from __future__ import annotations

from src.workflows.multitalk_prompt_builder import (
    FPS,
    NUM_FRAMES,
    VAE_NAME,
    build_multitalk_two_speaker_prompt,
)


def _body(**overrides):
    kwargs = dict(
        ref_image_filename="ref.png",
        speaker_1_audio_filename="s1.mp3",
        speaker_1_mask_filename="mask1.png",
        speaker_2_audio_filename="s2.mp3",
        speaker_2_mask_filename="mask2.png",
        positive_prompt="two people talking",
        seed=12345,
    )
    kwargs.update(overrides)
    return build_multitalk_two_speaker_prompt(**kwargs)


def test_build_multitalk_prompt_wires_each_speakers_real_audio_and_mask_separately():
    # Real finding from the live proof: audio_1/mask_1 must correspond to
    # the SAME speaker (speaker 1), never crossed - a swapped pairing
    # would drive speaker 1's mouth from speaker 2's real audio.
    body = _body()
    prompt = body["prompt"]
    assert prompt["5"]["inputs"]["audio"] == "s1.mp3"
    assert prompt["6"]["inputs"]["audio"] == "s2.mp3"
    assert prompt["15"]["inputs"]["image"] == "mask1.png"
    assert prompt["16"]["inputs"]["image"] == "mask2.png"
    assert prompt["7"]["inputs"]["audio_1"] == ["5", 0]
    assert prompt["7"]["inputs"]["audio_2"] == ["6", 0]
    assert prompt["17"]["inputs"]["mask_1"] == ["15", 0]
    assert prompt["17"]["inputs"]["mask_2"] == ["16", 0]


def test_build_multitalk_prompt_wires_the_real_required_ref_target_masks():
    # Real incident, 2026-10-23: the first real run crashed with
    # `'NoneType' object has no attribute 'max'` inside the real sampler
    # because ref_target_masks was never wired - contradicting this
    # project's own earlier (wrong) research note that "add" mode needed
    # no mask. Pin that the real fix is actually in the graph.
    body = _body()
    assert body["prompt"]["7"]["inputs"]["ref_target_masks"] == ["17", 0]
    assert body["prompt"]["17"]["class_type"] == "MaskBatchMulti"
    assert body["prompt"]["17"]["inputs"]["inputcount"] == 2


def test_build_multitalk_prompt_uses_sequential_add_mode_for_scripted_dialogue():
    # "add" (sequential turns) is correct for scripted back-and-forth
    # dialogue - "para" (simultaneous overlapping speech) would be wrong
    # for a screenplay where characters take turns.
    body = _body()
    assert body["prompt"]["7"]["inputs"]["multi_audio_type"] == "add"


def test_build_multitalk_prompt_real_model_filenames_match_the_live_proof():
    body = _body()
    prompt = body["prompt"]
    assert prompt["1"]["inputs"]["model_name"] == VAE_NAME == "Wan2_1_VAE_bf16.safetensors"
    assert prompt["2"]["inputs"]["model"] == "multitalk.safetensors"
    assert prompt["3"]["inputs"]["model"] == "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors"
    assert prompt["3"]["inputs"]["multitalk_model"] == ["2", 0]
    assert prompt["4"]["inputs"]["model"] == "chinese-wav2vec2-base.bin"


def test_build_multitalk_prompt_real_sampler_values_match_the_live_proof():
    body = _body(seed=99)
    sampler = body["prompt"]["11"]["inputs"]
    assert sampler["steps"] == 6
    assert sampler["cfg"] == 1.0
    assert sampler["shift"] == 11.0
    assert sampler["seed"] == 99
    assert sampler["scheduler"] == "dpm++_sde"
    assert sampler["multitalk_embeds"] == ["7", 0]


def test_build_multitalk_prompt_default_frames_and_fps_match_the_live_proof():
    body = _body()
    embeds = body["prompt"]["7"]["inputs"]
    assert embeds["num_frames"] == NUM_FRAMES == 81
    assert embeds["fps"] == FPS == 25.0


def test_build_multitalk_prompt_real_image_to_video_node_uses_the_ref_image_and_vae():
    body = _body(ref_image_filename="two_people.png")
    i2v = body["prompt"]["9"]["inputs"]
    assert i2v["start_image"] == ["8", 0]
    assert i2v["vae"] == ["1", 0]
    assert body["prompt"]["8"]["inputs"]["image"] == "two_people.png"
    assert i2v["width"] == 832
    assert i2v["height"] == 480


def test_build_multitalk_prompt_passes_the_real_positive_prompt_through():
    body = _body(positive_prompt="a man and a woman arguing in a kitchen")
    assert body["prompt"]["10"]["inputs"]["positive_prompt"] == "a man and a woman arguing in a kitchen"


def test_build_multitalk_prompt_output_video_carries_the_real_multitalk_audio():
    # CreateVideo's own audio must come from node "7" (MultiTalkWav2VecEmbeds'
    # own combined audio output), not a separately re-mixed track - the
    # same real pairing proven live.
    body = _body()
    assert body["prompt"]["13"]["inputs"]["audio"] == ["7", 1]
    assert body["prompt"]["13"]["inputs"]["images"] == ["12", 0]
