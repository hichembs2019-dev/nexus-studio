"""Builds the exact ComfyUI API-format prompt for a Wan 2.2 TI2V-5B text-
to-video generation. Every node id, input name and link below was hand-
verified against a real ComfyUI run on 2026-10-17 (see docs/SOURCES.md
"Wan 2.2 TI2V-5B real generation"): submitted via POST /prompt, confirmed
with no node_errors, watched run on the real GPU via /queue, and confirmed
"status: success" with a real, playable output file via /history and
/view. The start_image input on node 55 is genuinely optional (confirmed
via /object_info before dropping it) - this is pure text-to-video, no
reference image needed.

The official Comfy-Org template names umt5_xxl_fp8_e4m3fn_scaled.safetensors
for node 38's clip_name; the real run used umt5_xxl_fp16.safetensors instead
because that is the file actually downloaded (same text encoder, full
precision, listed in the same Comfy-Org/Wan_2.2_ComfyUI_Repackaged repo) -
a deliberate substitution, not a guess, and the one this module defaults to.
"""
from __future__ import annotations

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

UNET_NAME = "wan2.2_ti2v_5B_fp16.safetensors"
CLIP_NAME = "umt5_xxl_fp16.safetensors"
VAE_NAME = "wan2.2_vae.safetensors"


def build_ti2v_5b_prompt(
    *,
    positive_prompt: str,
    seed: int,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    width: int = 1280,
    height: int = 704,
    length: int = 121,
    steps: int = 20,
    cfg: float = 5,
    fps: float = 24,
    filename_prefix: str = "nexus_studio",
    client_id: str = "nexus-studio",
) -> dict:
    """Returns the full {"client_id": ..., "prompt": {...}} body for POST
    /prompt - the exact graph shape verified live (node ids kept identical
    to the real run so a diff against docs/SOURCES.md stays meaningful).
    """
    return {
        "client_id": client_id,
        "prompt": {
            "37": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"}},
            "38": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP_NAME, "type": "wan", "device": "default"}},
            "39": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
            "48": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["37", 0], "shift": 8}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": positive_prompt, "clip": ["38", 0]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["38", 0]}},
            "55": {
                "class_type": "Wan22ImageToVideoLatent",
                "inputs": {"vae": ["39", 0], "width": width, "height": height, "length": length, "batch_size": 1},
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["48", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["55", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "uni_pc",
                    "scheduler": "simple",
                    "denoise": 1,
                },
            },
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}},
            "57": {"class_type": "CreateVideo", "inputs": {"images": ["8", 0], "fps": fps}},
            "58": {
                "class_type": "SaveVideo",
                "inputs": {"video": ["57", 0], "filename_prefix": filename_prefix, "format": "auto", "codec": "auto"},
            },
        },
    }
