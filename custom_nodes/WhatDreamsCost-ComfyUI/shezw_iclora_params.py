import logging

import comfy.utils
import comfy_extras.nodes_lt as nodes_lt
import node_helpers
import torch
from comfy_api.latest import io

GuideData = io.Custom("GUIDE_DATA")
log = logging.getLogger(__name__)


def _primary_control(guide_data, default_frame_idx=0, default_strength=0.0):
    controls = []
    if isinstance(guide_data, dict):
        controls = guide_data.get("controls", []) or []

    usable = []
    for seg in controls:
        try:
            length = int(seg.get("length", 0))
            strength = float(seg.get("strength", default_strength))
            start = int(seg.get("start", default_frame_idx))
        except Exception:
            continue
        if length <= 0 or strength <= 0:
            continue
        usable.append((start, strength, seg.get("type", "union_control")))

    if not usable:
        return int(default_frame_idx), float(default_strength), ""

    usable.sort(key=lambda item: item[0])
    frame_idx, strength, control_type = usable[0]
    return int(frame_idx), float(strength), str(control_type)


def _append_guide_attention_entry(conditioning, pre_filter_count, latent_shape, attention_strength=1.0):
    existing_entries = []
    for t in conditioning:
        entries = t[1].get("guide_attention_entries", None)
        if entries is not None:
            existing_entries = entries
            break

    entries = [*existing_entries]
    entries.append({
        "pre_filter_count": pre_filter_count,
        "strength": attention_strength,
        "pixel_mask": None,
        "latent_shape": latent_shape,
    })
    return node_helpers.conditioning_set_values(conditioning, {"guide_attention_entries": entries})


def _dilate_latent(latent, horizontal_scale, vertical_scale):
    if horizontal_scale == 1 and vertical_scale == 1:
        return latent

    samples = latent["samples"]
    mask = latent.get("noise_mask", None)
    dilated_shape = samples.shape[:3] + (
        samples.shape[3] * vertical_scale,
        samples.shape[4] * horizontal_scale,
    )

    dilated_samples = torch.zeros(
        dilated_shape,
        device=samples.device,
        dtype=samples.dtype,
        requires_grad=False,
    )
    dilated_samples[..., ::vertical_scale, ::horizontal_scale] = samples

    dilated_mask_shape = (
        dilated_samples.shape[0],
        1,
        dilated_samples.shape[2],
        dilated_samples.shape[3],
        dilated_samples.shape[4],
    )
    dilated_mask = torch.full(
        dilated_mask_shape,
        -1.0,
        device=samples.device,
        dtype=samples.dtype,
        requires_grad=False,
    )
    dilated_mask[..., ::vertical_scale, ::horizontal_scale] = mask if mask is not None else 1.0
    return {"samples": dilated_samples, "noise_mask": dilated_mask}


class ShezwDirectorICLoRAParams(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ShezwDirectorICLoRAParams",
            display_name="Shezw Director IC-LoRA Params",
            category="WhatDreamsCost/LTX Director",
            description="Extracts the primary IC-Control segment from LTX Director guide_data for real IC-LoRA guide nodes.",
            inputs=[
                GuideData.Input("guide_data"),
                io.Int.Input("default_frame_idx", default=0, min=-9999, max=9999),
                io.Float.Input("default_strength", default=0.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Int.Output("frame_idx"),
                io.Float.Output("strength"),
                io.String.Output("control_type"),
            ],
        )

    @classmethod
    def execute(cls, guide_data, default_frame_idx=0, default_strength=0.0) -> io.NodeOutput:
        frame_idx, strength, control_type = _primary_control(guide_data, default_frame_idx, default_strength)
        return io.NodeOutput(frame_idx, strength, control_type)


class ShezwDirectorICLoRAGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ShezwDirectorICLoRAGuide",
            display_name="Shezw Director IC-LoRA Guide",
            category="WhatDreamsCost/LTX Director",
            description="Applies real LTX IC-LoRA guide conditioning using Director IC-Control track timing and strength.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                GuideData.Input("guide_data"),
                io.Image.Input("control_image", optional=True),
                io.Float.Input("latent_downscale_factor", default=1.0, min=1.0, max=10.0, step=1.0),
                io.Float.Input("default_strength", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("crop", options=["disabled", "center"], default="disabled"),
                io.Boolean.Input("use_tiled_encode", default=False),
                io.Int.Input("tile_size", default=256, min=64, max=512, step=32),
                io.Int.Input("tile_overlap", default=64, min=16, max=256, step=16),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def _encode(cls, vae, latent_width, latent_height, images, scale_factors,
                latent_downscale_factor, crop, use_tiled_encode, tile_size, tile_overlap):
        time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
        num_frames_to_keep = ((images.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
        images = images[:num_frames_to_keep]
        target_width = int(latent_width * width_scale_factor / latent_downscale_factor)
        target_height = int(latent_height * height_scale_factor / latent_downscale_factor)
        pixels = comfy.utils.common_upscale(
            images.movedim(-1, 1),
            target_width,
            target_height,
            "bilinear",
            crop=crop,
        ).movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        if use_tiled_encode:
            guide_latent = vae.encode_tiled(
                encode_pixels,
                tile_x=tile_size,
                tile_y=tile_size,
                overlap=tile_overlap,
            )
        else:
            guide_latent = vae.encode(encode_pixels)
        return encode_pixels, guide_latent

    @classmethod
    def execute(cls, positive, negative, vae, latent, guide_data, control_image=None,
                latent_downscale_factor=1.0, default_strength=0.0, crop="disabled",
                use_tiled_encode=False, tile_size=256, tile_overlap=64) -> io.NodeOutput:
        frame_idx, strength, _control_type = _primary_control(guide_data, 0, default_strength)
        if control_image is None or strength <= 0:
            return io.NodeOutput(positive, negative, latent)

        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = nodes_lt.get_noise_mask(latent)
        _, _, latent_length, latent_height, latent_width = latent_image.shape

        time_scale_factor = scale_factors[0]
        num_frames_to_keep = ((control_image.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
        causal_fix = frame_idx == 0 or num_frames_to_keep == 1
        if not causal_fix:
            control_image = torch.cat([control_image[:1], control_image], dim=0)

        control_image, guide_latent = cls._encode(
            vae,
            latent_width,
            latent_height,
            control_image,
            scale_factors,
            latent_downscale_factor,
            crop,
            use_tiled_encode,
            tile_size,
            tile_overlap,
        )

        if not causal_fix:
            guide_latent = guide_latent[:, :, 1:, :, :]
            control_image = control_image[1:]

        guide_orig_shape = list(guide_latent.shape[2:])
        guide_mask = None

        if latent_downscale_factor > 1:
            if latent_width % latent_downscale_factor != 0 or latent_height % latent_downscale_factor != 0:
                raise ValueError(
                    f"Latent spatial size {latent_width}x{latent_height} must be divisible by "
                    f"latent_downscale_factor {latent_downscale_factor}"
                )
            dilated = _dilate_latent(
                {"samples": guide_latent},
                horizontal_scale=int(latent_downscale_factor),
                vertical_scale=int(latent_downscale_factor),
            )
            guide_mask = dilated["noise_mask"]
            guide_latent = dilated["samples"]

        iclora_tokens_added = guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
        frame_idx, latent_idx = nodes_lt.LTXVAddGuide.get_latent_index(
            positive, latent_length, len(control_image), frame_idx, scale_factors
        )
        assert latent_idx + guide_latent.shape[2] <= latent_length, (
            "IC-LoRA control frames exceed the length of the latent sequence."
        )

        positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
            positive,
            negative,
            frame_idx,
            latent_image,
            noise_mask,
            guide_latent,
            strength,
            scale_factors,
            guide_mask=guide_mask,
            latent_downscale_factor=latent_downscale_factor,
            causal_fix=causal_fix,
        )

        positive = _append_guide_attention_entry(positive, iclora_tokens_added, guide_orig_shape)
        negative = _append_guide_attention_entry(negative, iclora_tokens_added, guide_orig_shape)
        log.info("[ShezwDirectorICLoRAGuide] Applied IC-LoRA guide at frame %s strength %.3f", frame_idx, strength)
        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})
