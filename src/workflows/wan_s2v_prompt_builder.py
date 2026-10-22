"""Builds the ComfyUI API-format prompt for a self-hosted Wan 2.2 S2V
(Speech-to-Video) generation: one reference photo + one real audio clip
(a character's TTS line) in, one real lip-synced video clip out.

This is the self-hosted replacement for Replicate's wan-video/wan-2.2-s2v
(see replicate_video_client.py) - same real model family, run on the
GPU pod this project already owns instead of paying Replicate per call.

Real history, not guessed: a first version of this graph (built from
`/object_info` node schemas alone, 2026-10-22) was REJECTED by a real
run - no node_errors at submit time, but KSampler crashed with a real
tensor-shape mismatch ("size of tensor a (48) must match tensor b (16)
at dimension 1"). Root cause, found by fetching Comfy-Org's own official
template (`Comfy-Org/workflow_templates`, `video_wan2_2_14B_s2v.json`)
and diffing it against the first attempt: the VAE was wrong
(`wan2.2_vae.safetensors`, the TI2V-5B VAE, carried over from
wan_prompt_builder.py by assumption) - S2V needs `wan_2.1_vae.safetensors`
instead, a DIFFERENT real file from a DIFFERENT model generation. Every
value below is now read directly from that official template's own real
node graph and widget values, not re-guessed:
  - ModelSamplingSD3.shift = 8 (not this node's own schema default of 3.0)
  - KSampler steps=4, cfg=1.0, sampler=uni_pc, scheduler=simple - the
    official template's ENABLED (mode=0) path runs the 4-step lightning
    LoRA (wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors);
    its alternate 20-step/cfg-6 no-LoRA path exists in the same template
    but is left disabled by its own authors. The LoRA path was used here
    since it is the template's own real default, not a cost-cutting
    shortcut invented here.
  - WanSoundImageToVideo width=height=640 (the template's own widget
    values - this node does not expose them as linkable inputs, only
    `length` is overridden via a connected value in the template, which
    is why `length` alone is computed dynamically below)
  - CreateVideo fps=16, with the SAME real audio file wired into its own
    `audio` input (not just AudioEncoderEncode's) - confirmed via the
    official template's own links array: LoadAudio feeds both
    AudioEncoderEncode AND CreateVideo.

Real model files (Comfy-Org/Wan_2.2_ComfyUI_Repackaged, the same
official repo that supplied the TI2V-5B set - every size confirmed live
via `curl -sIL` before downloading, see docs/SOURCES.md):
  wan2.2_s2v_14B_fp8_scaled.safetensors                    16,394,832,474 bytes
  umt5_xxl_fp16.safetensors                                 11,366,399,385 bytes (shared with TI2V-5B)
  wan_2.1_vae.safetensors                                      253,815,318 bytes (NOT wan2.2_vae.safetensors)
  wav2vec2_large_english_fp16.safetensors                      630,997,322 bytes
  wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors 1,226,977,424 bytes

Deliberately NOT yet included: the official template's own "double the
first frame, decode, then cut it back out" VAE-artifact fix (its own
Note calls the first frame "overbaked" by the VAE without it). That is
a real quality fix, not a correctness one - omitted here to keep this
first real graph provably correct end to end first, with the quality
fix a tracked follow-up once a real clip has been produced and judged.
"""
from __future__ import annotations

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

UNET_NAME = "wan2.2_s2v_14B_fp8_scaled.safetensors"
CLIP_NAME = "umt5_xxl_fp16.safetensors"
VAE_NAME = "wan_2.1_vae.safetensors"
AUDIO_ENCODER_NAME = "wav2vec2_large_english_fp16.safetensors"
LORA_NAME = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"

FPS = 16.0
# The official template's own Note: "This model needs at least 73 frames."
MIN_FRAMES = 73


def frames_for_duration(duration_seconds: float, *, fps: float = FPS) -> int:
    """WanSoundImageToVideo's `length` accepts step=4 increments (its own
    real default, 77, fits the 4n+1 shape) and the model needs at least
    73 real frames regardless of how short the real audio is. Always
    rounds UP, never fewer frames than the real audio needs, so a line
    is never cut short."""
    raw_frames = max(duration_seconds * fps, MIN_FRAMES)
    n = -(-(int(raw_frames) - 1) // 4)  # ceil((raw_frames - 1) / 4)
    return max(MIN_FRAMES, 4 * n + 1)


def build_s2v_prompt(
    *,
    positive_prompt: str,
    ref_image_filename: str,
    audio_filename: str,
    audio_duration_seconds: float,
    seed: int,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    width: int = 640,
    height: int = 640,
    steps: int = 4,
    cfg: float = 1.0,
    fps: float = FPS,
    filename_prefix: str = "nexus_studio_s2v",
    client_id: str = "nexus-studio",
) -> dict:
    """Returns the full {"client_id": ..., "prompt": {...}} body for
    POST /prompt. `ref_image_filename`/`audio_filename` must already be
    uploaded to the pod via POST /upload/image (works for any file type
    despite the name - confirmed live 2026-10-22) and are referenced
    here by the filename that upload returns.
    """
    length = frames_for_duration(audio_duration_seconds, fps=fps)
    return {
        "client_id": client_id,
        "prompt": {
            "37": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"}},
            "107": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": ["37", 0], "lora_name": LORA_NAME, "strength_model": 1},
            },
            "48": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["107", 0], "shift": 8}},
            "38": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP_NAME, "type": "wan", "device": "default"}},
            "39": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
            "60": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": AUDIO_ENCODER_NAME}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": positive_prompt, "clip": ["38", 0]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["38", 0]}},
            "61": {"class_type": "LoadImage", "inputs": {"image": ref_image_filename}},
            "62": {"class_type": "LoadAudio", "inputs": {"audio": audio_filename}},
            "63": {"class_type": "AudioEncoderEncode", "inputs": {"audio_encoder": ["60", 0], "audio": ["62", 0]}},
            "55": {
                "class_type": "WanSoundImageToVideo",
                "inputs": {
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "vae": ["39", 0],
                    "width": width,
                    "height": height,
                    "length": length,
                    "batch_size": 1,
                    "audio_encoder_output": ["63", 0],
                    "ref_image": ["61", 0],
                },
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["48", 0],
                    "positive": ["55", 0],
                    "negative": ["55", 1],
                    "latent_image": ["55", 2],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "uni_pc",
                    "scheduler": "simple",
                    "denoise": 1,
                },
            },
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}},
            "57": {"class_type": "CreateVideo", "inputs": {"images": ["8", 0], "fps": fps, "audio": ["62", 0]}},
            "58": {
                "class_type": "SaveVideo",
                "inputs": {"video": ["57", 0], "filename_prefix": filename_prefix, "format": "auto", "codec": "auto"},
            },
        },
    }
