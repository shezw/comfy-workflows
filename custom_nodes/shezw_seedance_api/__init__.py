import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime

try:
    import folder_paths
except Exception:
    folder_paths = None


FAL_REFERENCE_ENDPOINT = "bytedance/seedance-2.0/reference-to-video"
FAL_REFERENCE_FAST_ENDPOINT = "bytedance/seedance-2.0/fast/reference-to-video"


def _output_dir():
    if folder_paths is not None:
        return folder_paths.get_output_directory()
    return os.getcwd()


def _json_request(method, url, api_key=None, payload=None, timeout=60):
    data = None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Key {api_key}"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e


def _download_file(url, filename_prefix):
    out_dir = _output_dir()
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"{filename_prefix}_{stamp}.mp4")
    req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Shezw-Seedance/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        with open(path, "wb") as f:
            f.write(resp.read())
    return path


def _image_to_data_uri(image, mime="image/png"):
    if image is None:
        return None
    import numpy as np
    from PIL import Image

    arr = image[0].detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _audio_to_wav_data_uri(audio):
    if audio is None:
        return None
    import numpy as np

    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 44100))
    if waveform is None:
        return None
    wav = waveform.detach().cpu().float().numpy()
    # Comfy AUDIO is usually [batch, channels, samples].
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim == 1:
        wav = wav[None, :]
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav.T * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return f"data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _clean(text):
    return " ".join(str(text or "").strip().split())


class SD2_DirectorPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_prompt": ("STRING", {"multiline": True, "default": "cinematic 3D animated scene, detailed materials, soft production lighting"}),
                "character_prompt": ("STRING", {"multiline": True, "default": "keep the same character identity, face, outfit, material details and art direction as @Image1"}),
                "action_prompt": ("STRING", {"multiline": True, "default": "the character speaks naturally, accurate lip sync, subtle head nods, restrained hand gestures"}),
                "camera_prompt": ("STRING", {"multiline": True, "default": "locked stable medium shot, eye level, no camera shake"}),
                "timeline_json": ("STRING", {"multiline": True, "default": "[{\"start\":0,\"end\":2,\"action\":\"begins speaking with a small nod\"},{\"start\":2,\"end\":5,\"action\":\"right hand makes a small open palm gesture while speaking\"}]"}),
                "reference_instruction": ("STRING", {"multiline": True, "default": "Use @Image1 as the main identity and scene reference. Use @Image2-@Image9 only when present for clothing, material texture, pose, lighting, and background consistency. If @Audio1 is present, align mouth movement and speaking performance to it."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("seedance_prompt", "director_manifest")
    FUNCTION = "compile"
    CATEGORY = "shezw/seedance2-api"

    def compile(self, scene_prompt, character_prompt, action_prompt, camera_prompt, timeline_json, reference_instruction):
        timeline = []
        try:
            parsed = json.loads(timeline_json) if timeline_json.strip() else []
            if isinstance(parsed, list):
                timeline = parsed
        except Exception:
            timeline = []

        timeline_parts = []
        for i, seg in enumerate(timeline, start=1):
            if not isinstance(seg, dict):
                continue
            start = seg.get("start", "")
            end = seg.get("end", "")
            action = _clean(seg.get("action") or seg.get("prompt") or seg.get("motion") or "")
            camera = _clean(seg.get("camera", ""))
            speech = _clean(seg.get("speech", ""))
            part = f"Segment {i} ({start}s-{end}s): {action}"
            if speech:
                part += f"; spoken content meaning: {speech}"
            if camera:
                part += f"; camera: {camera}"
            timeline_parts.append(part)

        prompt = "\n".join([
            "Director-level Seedance 2.0 reference-to-video generation.",
            f"Scene: {_clean(scene_prompt)}",
            f"Character and identity lock: {_clean(character_prompt)}",
            f"Core action performance: {_clean(action_prompt)}",
            f"Camera: {_clean(camera_prompt)}",
            f"References: {_clean(reference_instruction)}",
            "Timeline actions, follow in order:",
            "\n".join(timeline_parts) if timeline_parts else "Maintain natural speaking motion and stable performance for the full clip.",
            "Preserve all important reference identity, materials, fabric texture, props, background layout, and lighting. Avoid identity drift, jitter, warped hands, mouth hidden by hands, broken lip sync, text, subtitles, logos, and watermark.",
        ])

        manifest = json.dumps({
            "scene_prompt": scene_prompt,
            "character_prompt": character_prompt,
            "action_prompt": action_prompt,
            "camera_prompt": camera_prompt,
            "timeline": timeline,
            "reference_instruction": reference_instruction,
        }, ensure_ascii=False, indent=2)
        return (prompt, manifest)


class Seedance2FalGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "endpoint": ([
                    FAL_REFERENCE_ENDPOINT,
                    FAL_REFERENCE_FAST_ENDPOINT,
                    "bytedance/seedance-2.0/text-to-video",
                    "bytedance/seedance-2.0/fast/text-to-video",
                    "bytedance/seedance-2.0/image-to-video",
                    "bytedance/seedance-2.0/fast/image-to-video",
                ], {"default": FAL_REFERENCE_ENDPOINT}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "duration": (["auto", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"], {"default": "5"}),
                "aspect_ratio": (["auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"], {"default": "16:9"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "wait_for_result": ("BOOLEAN", {"default": True}),
                "poll_interval_seconds": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 30.0, "step": 0.5}),
                "timeout_seconds": ("INT", {"default": 900, "min": 30, "max": 7200}),
                "filename_prefix": ("STRING", {"default": "seedance2_director"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "extra_json": ("STRING", {"multiline": True, "default": "{}"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "local_video_path", "request_id", "response_json")
    FUNCTION = "generate"
    CATEGORY = "shezw/seedance2-api"

    def generate(self, prompt, api_key, endpoint, resolution, duration, aspect_ratio, generate_audio,
                 seed, wait_for_result, poll_interval_seconds, timeout_seconds, filename_prefix,
                 extra_json="{}", **kwargs):
        key = api_key.strip() or os.environ.get("FAL_KEY", "").strip()
        if not key:
            raise RuntimeError("Missing fal API key. Set FAL_KEY in the environment or fill api_key on the node.")

        image_urls = []
        for i in range(1, 10):
            uri = _image_to_data_uri(kwargs.get(f"image_{i}"))
            if uri:
                image_urls.append(uri)

        audio_urls = []
        for i in range(1, 4):
            uri = _audio_to_wav_data_uri(kwargs.get(f"audio_{i}"))
            if uri:
                audio_urls.append(uri)

        payload = {
            "prompt": prompt,
            "resolution": resolution,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "generate_audio": bool(generate_audio),
        }
        if seed >= 0:
            payload["seed"] = int(seed)
        if image_urls:
            payload["image_urls"] = image_urls
        if audio_urls:
            payload["audio_urls"] = audio_urls

        try:
            extra = json.loads(extra_json) if extra_json and extra_json.strip() else {}
            if isinstance(extra, dict):
                payload.update(extra)
        except Exception as e:
            raise RuntimeError(f"extra_json is not valid JSON: {e}") from e

        submit_url = f"https://queue.fal.run/{endpoint}"
        submit = _json_request("POST", submit_url, key, payload, timeout=120)
        request_id = submit.get("request_id", "")
        if not wait_for_result:
            return ("", "", request_id, json.dumps({"submitted": submit, "payload": _redact_payload(payload)}, ensure_ascii=False, indent=2))

        status_url = submit.get("status_url") or f"https://queue.fal.run/{endpoint}/requests/{request_id}/status"
        response_url = submit.get("response_url") or f"https://queue.fal.run/{endpoint}/requests/{request_id}/response"
        start = time.time()
        status = {}
        while True:
            if time.time() - start > timeout_seconds:
                raise RuntimeError(f"Seedance request timed out after {timeout_seconds}s. request_id={request_id}")
            status = _json_request("GET", status_url + ("&logs=1" if "?" in status_url else "?logs=1"), key, None, timeout=60)
            if status.get("status") == "COMPLETED":
                if status.get("error"):
                    raise RuntimeError(f"Seedance request failed: {status.get('error')}")
                break
            if status.get("status") in ("FAILED", "CANCELLED", "ERROR"):
                raise RuntimeError(f"Seedance request failed: {json.dumps(status, ensure_ascii=False)}")
            if status.get("error"):
                raise RuntimeError(f"Seedance request failed: {status.get('error')}")
            time.sleep(float(poll_interval_seconds))

        result = _json_request("GET", response_url, key, None, timeout=120)
        video = result.get("video") or {}
        video_url = video.get("url") or result.get("video_url") or ""
        local_path = ""
        if video_url:
            local_path = _download_file(video_url, filename_prefix)

        response = {
            "request_id": request_id,
            "status": status,
            "result": result,
            "payload": _redact_payload(payload),
        }
        return (video_url, local_path, request_id, json.dumps(response, ensure_ascii=False, indent=2))


def _redact_payload(payload):
    redacted = dict(payload)
    for key in ("image_urls", "audio_urls", "video_urls"):
        if key in redacted:
            redacted[key] = [f"<{key[:-1]}:{i+1} data-uri/url>" for i, _ in enumerate(redacted[key])]
    return redacted


NODE_CLASS_MAPPINGS = {
    "SD2_DirectorPrompt": SD2_DirectorPrompt,
    "Seedance2FalGenerate": Seedance2FalGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SD2_DirectorPrompt": "SD2 Director Prompt",
    "Seedance2FalGenerate": "Seedance 2.0 fal Generate",
}
