from __future__ import annotations

from src.workflows.wan_prompt_builder import DEFAULT_NEGATIVE_PROMPT, build_ti2v_5b_prompt


def test_build_ti2v_5b_prompt_matches_the_real_verified_graph():
    body = build_ti2v_5b_prompt(positive_prompt="a red car on a coastal road", seed=42)

    prompt = body["prompt"]
    assert body["client_id"] == "nexus-studio"
    assert prompt["37"] == {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": "default"}}
    assert prompt["38"]["inputs"]["clip_name"] == "umt5_xxl_fp16.safetensors"
    assert prompt["39"] == {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}}
    assert prompt["6"]["inputs"] == {"text": "a red car on a coastal road", "clip": ["38", 0]}
    assert prompt["7"]["inputs"] == {"text": DEFAULT_NEGATIVE_PROMPT, "clip": ["38", 0]}
    assert prompt["55"]["inputs"] == {"vae": ["39", 0], "width": 1280, "height": 704, "length": 121, "batch_size": 1}
    assert "start_image" not in prompt["55"]["inputs"]
    assert prompt["3"]["inputs"]["seed"] == 42
    assert prompt["3"]["inputs"]["latent_image"] == ["55", 0]
    assert prompt["8"]["inputs"] == {"samples": ["3", 0], "vae": ["39", 0]}
    assert prompt["57"]["inputs"] == {"images": ["8", 0], "fps": 24}
    assert prompt["58"]["inputs"]["filename_prefix"] == "nexus_studio"


def test_build_ti2v_5b_prompt_overrides_filename_prefix_and_dimensions():
    body = build_ti2v_5b_prompt(positive_prompt="x", seed=1, filename_prefix="episode1_scene002", width=960, height=544)

    prompt = body["prompt"]
    assert prompt["58"]["inputs"]["filename_prefix"] == "episode1_scene002"
    assert prompt["55"]["inputs"]["width"] == 960
    assert prompt["55"]["inputs"]["height"] == 544
