from __future__ import annotations

from src.workflows.wan_s2v_prompt_builder import (
    AUDIO_ENCODER_NAME,
    CLIP_NAME,
    LORA_NAME,
    MIN_FRAMES,
    UNET_NAME,
    VAE_NAME,
    build_s2v_prompt,
    frames_for_duration,
)


def test_frames_for_duration_enforces_the_real_minimum_even_for_a_very_short_line():
    # Real constraint from the official template: "This model needs at
    # least 73 frames." A one-word line must not produce fewer.
    assert frames_for_duration(0.5) == MIN_FRAMES


def test_frames_for_duration_rounds_up_to_cover_the_real_audio_never_short():
    # 10s at 16fps = 160 raw frames - must round UP to the next valid
    # 4n+1 value, never down (never cut the real line short).
    frames = frames_for_duration(10.0)
    assert frames >= 10.0 * 16
    assert (frames - 1) % 4 == 0


def test_frames_for_duration_real_5_232s_clip_matches_the_live_proof():
    # The real value submitted and proven live 2026-10-22 (pod
    # wt5ennaph7w5nh, prompt bfd49060-...): 85 frames for a 5.232s clip.
    assert frames_for_duration(5.232) == 85


def test_build_s2v_prompt_uses_the_real_vae_not_the_ti2v_5b_one():
    # Mutation check built into the test itself: the first real attempt
    # at this graph used wan2.2_vae.safetensors (the TI2V-5B VAE) and a
    # real submission crashed inside KSampler with a tensor-shape
    # mismatch. S2V needs wan_2.1_vae.safetensors - a different file
    # from a different model generation entirely.
    body = build_s2v_prompt(
        positive_prompt="p", ref_image_filename="ref.png", audio_filename="a.mp3",
        audio_duration_seconds=5.0, seed=1,
    )
    assert body["prompt"]["39"]["inputs"]["vae_name"] == VAE_NAME
    assert VAE_NAME == "wan_2.1_vae.safetensors"
    assert VAE_NAME != "wan2.2_vae.safetensors"


def test_build_s2v_prompt_wires_the_lightning_lora_before_model_sampling():
    body = build_s2v_prompt(
        positive_prompt="p", ref_image_filename="ref.png", audio_filename="a.mp3",
        audio_duration_seconds=5.0, seed=1,
    )
    prompt = body["prompt"]
    assert prompt["107"]["class_type"] == "LoraLoaderModelOnly"
    assert prompt["107"]["inputs"]["lora_name"] == LORA_NAME
    assert prompt["107"]["inputs"]["model"] == ["37", 0]
    assert prompt["48"]["inputs"]["model"] == ["107", 0]
    assert prompt["48"]["inputs"]["shift"] == 8


def test_build_s2v_prompt_real_ksampler_values_match_the_official_lightning_path():
    # Real values read from Comfy-Org's own official template's ENABLED
    # (non-muted) node graph, not this node's own schema defaults.
    body = build_s2v_prompt(
        positive_prompt="p", ref_image_filename="ref.png", audio_filename="a.mp3",
        audio_duration_seconds=5.0, seed=99,
    )
    ksampler = body["prompt"]["3"]["inputs"]
    assert ksampler["steps"] == 4
    assert ksampler["cfg"] == 1.0
    assert ksampler["seed"] == 99
    assert ksampler["sampler_name"] == "uni_pc"
    assert ksampler["scheduler"] == "simple"


def test_build_s2v_prompt_feeds_the_same_real_audio_to_both_encoder_and_video():
    # Real finding from the official template's own links: LoadAudio
    # feeds BOTH AudioEncoderEncode (drives the lip-sync) AND
    # CreateVideo's own audio input (the output file's real soundtrack).
    # Missing the second link would produce a silent "lip-synced" video.
    body = build_s2v_prompt(
        positive_prompt="p", ref_image_filename="ref.png", audio_filename="line.mp3",
        audio_duration_seconds=5.0, seed=1,
    )
    prompt = body["prompt"]
    assert prompt["62"]["inputs"]["audio"] == "line.mp3"
    assert prompt["63"]["inputs"]["audio"] == ["62", 0]
    assert prompt["57"]["inputs"]["audio"] == ["62", 0]


def test_build_s2v_prompt_real_model_filenames():
    body = build_s2v_prompt(
        positive_prompt="p", ref_image_filename="ref.png", audio_filename="a.mp3",
        audio_duration_seconds=5.0, seed=1,
    )
    prompt = body["prompt"]
    assert prompt["37"]["inputs"]["unet_name"] == UNET_NAME == "wan2.2_s2v_14B_fp8_scaled.safetensors"
    assert prompt["38"]["inputs"]["clip_name"] == CLIP_NAME == "umt5_xxl_fp16.safetensors"
    assert prompt["60"]["inputs"]["audio_encoder_name"] == AUDIO_ENCODER_NAME == "wav2vec2_large_english_fp16.safetensors"


def test_build_s2v_prompt_real_width_height_match_the_official_template():
    body = build_s2v_prompt(
        positive_prompt="p", ref_image_filename="ref.png", audio_filename="a.mp3",
        audio_duration_seconds=5.0, seed=1,
    )
    s2v_node = body["prompt"]["55"]["inputs"]
    assert s2v_node["width"] == 640
    assert s2v_node["height"] == 640
    assert s2v_node["ref_image"] == ["61", 0]
    assert s2v_node["audio_encoder_output"] == ["63", 0]
