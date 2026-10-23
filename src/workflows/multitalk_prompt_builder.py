"""Builds the ComfyUI API-format prompt for a self-hosted MultiTalk
two-speaker generation: one composite reference image + one per-speaker
face mask + one real TTS audio clip per speaker in, one real video with
both speakers sharing the same scene out.

Real history, not guessed - every node/widget value here is read directly
from a real, proven run on this project's own pod (`fdhxh9x1igrsih`,
2026-10-23), not from the author's own example workflow alone:

- The graph itself (`kijai/ComfyUI-WanVideoWrapper`'s MultiTalk nodes,
  traced from `example_workflows/wanvideo_2_1_14B_I2V_InfiniteTalk_
  example_03.json`) first crashed inside `MultiTalkWav2VecEmbeds` with
  `'NoneType' object is not subscriptable` - the installed
  `transformers==5.17.0`'s real `Wav2Vec2Encoder.forward()` no longer
  collects per-layer hidden states at all (confirmed by reading the
  installed package's own source), which the vendored node's own
  `multitalk/wav2vec2.py` still assumed it would. Fixed ON THE POD
  (`register_forward_hook`/`register_forward_pre_hook` added to that
  vendored file) - that fix lives in the pod's own image/volume, not in
  this repository, and is a precondition for this graph to run at all.
- `ref_target_masks` (node "17" below) is a REAL, REQUIRED `MASK` input
  for any `human_num >= 2` generation - not optional for "add"
  (sequential) mode, contradicting this project's own earlier research
  note (docs/SOURCES.md, Stage 1) that assumed otherwise. Without it,
  `WanVideoSampler` crashes with `'NoneType' object has no attribute
  'max'` inside the real cross-attention code (`multitalk/multitalk.py`).
- Real, successful run (`nexus_studio_multitalk_test_00001_.mp4`, 832x480,
  25fps, 5.48s, both speakers' mouths independently moving across real
  frames, downloaded and personally viewed) used exactly the values
  below. See docs/SOURCES.md, "MultiTalk two-speaker generation proven
  real, end to end".

Deliberately scoped to EXACTLY TWO speakers for now, not the model's own
documented 4-speaker ceiling: only `human_num=2` has ever actually run on
this project's own infrastructure. `MaskBatchMulti`'s own schema (read
live via `/object_info`) accepts more than two `mask_N` inputs via
`inputcount`, so 3-4 speakers is architecturally the same shape - but
extending `build_multitalk_prompt` to accept more than 2 speakers before a
real 3-speaker run has ever been attempted would be exactly the kind of
unproven claim this project's standing rule forbids ("لا تقول يبدو ممكن،
يجب ان تتأكد" - don't say it seems possible, you must be certain).
"""
from __future__ import annotations

UNET_NAME = "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors"
MULTITALK_NAME = "multitalk.safetensors"
WAV2VEC_NAME = "chinese-wav2vec2-base.bin"
VAE_NAME = "Wan2_1_VAE_bf16.safetensors"
T5_NAME = "umt5-xxl-enc-fp8_e4m3fn.safetensors"

DEFAULT_NEGATIVE_PROMPT = (
    "bright tones, overexposed, static, blurred details, subtitles, worst quality, "
    "low quality, deformed"
)

WIDTH = 832
HEIGHT = 480
NUM_FRAMES = 81
FPS = 25.0


def build_multitalk_two_speaker_prompt(
    *,
    ref_image_filename: str,
    speaker_1_audio_filename: str,
    speaker_1_mask_filename: str,
    speaker_2_audio_filename: str,
    speaker_2_mask_filename: str,
    positive_prompt: str,
    seed: int,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    num_frames: int = NUM_FRAMES,
    fps: float = FPS,
    width: int = WIDTH,
    height: int = HEIGHT,
    filename_prefix: str = "nexus_studio_multitalk",
    client_id: str = "nexus-studio-multitalk",
) -> dict:
    """Returns the full {"client_id": ..., "prompt": {...}} body for
    POST /prompt. Every *_filename argument must already be uploaded to
    the pod (POST /upload/image, which accepts any file type despite the
    name - see comfyui_s2v_client.py) and is referenced here by the
    filename that upload returned.
    """
    return {
        "client_id": client_id,
        "prompt": {
            "1": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": VAE_NAME, "precision": "bf16"}},
            "2": {"class_type": "MultiTalkModelLoader", "inputs": {"model": MULTITALK_NAME}},
            "3": {
                "class_type": "WanVideoModelLoader",
                "inputs": {
                    "model": UNET_NAME, "base_precision": "fp16_fast", "quantization": "disabled",
                    "load_device": "offload_device", "attention_mode": "sdpa",
                    "multitalk_model": ["2", 0],
                },
            },
            "4": {"class_type": "Wav2VecModelLoader", "inputs": {"model": WAV2VEC_NAME, "base_precision": "fp16", "load_device": "main_device"}},
            "5": {"class_type": "LoadAudio", "inputs": {"audio": speaker_1_audio_filename}},
            "6": {"class_type": "LoadAudio", "inputs": {"audio": speaker_2_audio_filename}},
            "15": {"class_type": "LoadImageMask", "inputs": {"image": speaker_1_mask_filename, "channel": "red"}},
            "16": {"class_type": "LoadImageMask", "inputs": {"image": speaker_2_mask_filename, "channel": "red"}},
            "17": {"class_type": "MaskBatchMulti", "inputs": {"inputcount": 2, "mask_1": ["15", 0], "mask_2": ["16", 0]}},
            "7": {
                "class_type": "MultiTalkWav2VecEmbeds",
                "inputs": {
                    "wav2vec_model": ["4", 0], "audio_1": ["5", 0], "audio_2": ["6", 0],
                    "normalize_loudness": True, "num_frames": num_frames, "fps": fps,
                    "audio_scale": 1.0, "audio_cfg_scale": 1.0, "multi_audio_type": "add",
                    "ref_target_masks": ["17", 0],
                },
            },
            "8": {"class_type": "LoadImage", "inputs": {"image": ref_image_filename}},
            "9": {
                "class_type": "WanVideoImageToVideoMultiTalk",
                "inputs": {
                    "vae": ["1", 0], "width": width, "height": height, "frame_window_size": num_frames,
                    "motion_frame": 25, "force_offload": False, "colormatch": "disabled",
                    "start_image": ["8", 0], "mode": "auto",
                },
            },
            "10": {
                "class_type": "WanVideoTextEncodeCached",
                "inputs": {
                    "model_name": T5_NAME, "precision": "bf16", "positive_prompt": positive_prompt,
                    "negative_prompt": negative_prompt, "quantization": "disabled",
                    "use_disk_cache": True, "device": "gpu",
                },
            },
            "11": {
                "class_type": "WanVideoSampler",
                "inputs": {
                    "model": ["3", 0], "image_embeds": ["9", 0], "steps": 6, "cfg": 1.0, "shift": 11.0,
                    "seed": seed, "force_offload": True, "scheduler": "dpm++_sde", "riflex_freq_index": 0,
                    "text_embeds": ["10", 0], "multitalk_embeds": ["7", 0],
                },
            },
            "12": {"class_type": "WanVideoPassImagesFromSamples", "inputs": {"samples": ["11", 0]}},
            "13": {"class_type": "CreateVideo", "inputs": {"images": ["12", 0], "fps": fps, "audio": ["7", 1]}},
            "14": {
                "class_type": "SaveVideo",
                "inputs": {"video": ["13", 0], "filename_prefix": filename_prefix, "format": "auto", "codec": "auto"},
            },
        },
    }
