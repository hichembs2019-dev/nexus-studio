"""Cinematic shot mode, end to end through DialoguePipeline with mocks:
Gemini director -> Groq compiler -> flux-2-dev shot still (from the
character's portrait) -> speech-to-video on that still."""
from __future__ import annotations

import base64
import json
import subprocess

import pytest

from src.agents.shot_director import ShotDirectorAgent
from src.app import CINEMATIC_VIDEO_HEIGHT, CINEMATIC_VIDEO_WIDTH, build_app
from src.config import Settings
from src.integrations.multitalk_video_client import MockMultiTalkVideoClient
from src.integrations.reasoning_client import MockAgentReasoningClient, RealGeminiReasoningClient
from src.integrations.replicate_image_client import (
    FLUX2_DEV_ENDPOINT,
    FLUX2_DEV_PRICE_PER_MEGAPIXEL_USD_UNVERIFIED,
    MockReplicateImageClient,
    MockShotImageClient,
    RealFlux2ShotImageClient,
    ReplicateImageError,
    estimated_shot_image_usd,
    flux2_dev_cost_usd,
)
from src.integrations.replicate_video_client import MockReplicateVideoClient
from src.integrations.tts_client import MockTTSClient
from src.models import Character, DialogueLine, DialogueSceneInfo, DialogueScreenplay
from src.workflows.dialogue_pipeline import (
    CINEMATIC_STYLE_SUFFIX,
    PORTRAIT_STYLE_SUFFIX,
    DialoguePipeline,
    DialoguePipelineError,
)
from src.workflows.shot_prompt_compiler import ShotPromptCompiler

PRICE = FLUX2_DEV_PRICE_PER_MEGAPIXEL_USD_UNVERIFIED


def _jpeg(path, color: str = "blue", size: str = "16x16") -> bytes:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={size}", "-frames:v", "1", str(path)],
        capture_output=True, check=True,
    )
    with open(path, "rb") as f:
        return f.read()


def _mp4(path, seconds: float = 1.0) -> bytes:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=16:d={seconds}", "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )
    with open(path, "rb") as f:
        return f.read()


def _screenplay(n: int = 3) -> DialogueScreenplay:
    names = ["فوزية", "جويدة"]
    return DialogueScreenplay(
        title="العشاء",
        premise="فوزية وجويدة في المطبخ",
        characters=[
            Character(name="فوزية", gender="female", appearance="woman in her 50s"),
            Character(name="جويدة", gender="female", appearance="woman in her 20s"),
        ],
        lines=[DialogueLine(index=i, character=names[i % 2], line=f"سطر {i}", action="stirs the pot") for i in range(n)],
        scenes=[DialogueSceneInfo(index=0, location="old kitchen", mood="warm")],
    )


def _director_reply(shot_types: list[str]) -> str:
    return json.dumps([
        {
            "line_index": i, "shot_type": t, "camera_angle": "eye_level", "camera_movement": "slow_push_in",
            "lighting": "warm window light", "environment_motion": "steam rising", "gaze": "at the other woman",
        }
        for i, t in enumerate(shot_types)
    ])


def _compiler_reply(n: int) -> str:
    return json.dumps([
        {"line_index": i, "image_prompt": f"A woman in a kitchen, frame {i}", "video_prompt": f"She speaks, beat {i}"}
        for i in range(n)
    ])


def _pipeline(tmp_path, *, shot_types=("wide", "close_up", "medium"), video=None, shot_client=None, director_reply=None, **kwargs):
    n = len(shot_types)
    director = ShotDirectorAgent(MockAgentReasoningClient([director_reply or _director_reply(list(shot_types))]))
    compiler = ShotPromptCompiler(MockAgentReasoningClient([_compiler_reply(n)]))
    video = video or MockReplicateVideoClient(scripted_bytes=_mp4(tmp_path / "src.mp4"))
    shot_client = shot_client or MockShotImageClient(scripted_bytes=_jpeg(tmp_path / "shot.jpg", "red"))
    kwargs.setdefault("image_client", MockReplicateImageClient(scripted_bytes=_jpeg(tmp_path / "portrait.jpg")))
    pipeline = DialoguePipeline(
        video, MockTTSClient(), str(tmp_path / "out"),
        shot_director=director, shot_compiler=compiler, shot_image_client=shot_client,
        **kwargs,
    )
    return pipeline, video, shot_client


# --- DialoguePipeline ---

def test_each_spoken_line_is_animated_from_its_own_shot_still_with_the_compiled_prompt(tmp_path):
    pipeline, video, shot_client = _pipeline(tmp_path)
    shot_bytes = _jpeg(tmp_path / "shot.jpg", "red")
    portrait_bytes = _jpeg(tmp_path / "portrait.jpg")

    produced = pipeline.produce(_screenplay())

    assert len(produced.clips) == 3 and produced.failed_lines == []
    first = video.calls[0]
    assert base64.b64decode(first["image_data_uri"].split(",", 1)[1]) == shot_bytes
    assert first["prompt"].startswith("slow camera push-in, She speaks, beat 0.")
    assert CINEMATIC_STYLE_SUFFIX not in first["prompt"]
    prompt, refs = shot_client.calls[0]
    assert prompt.startswith("wide establishing shot, eye-level camera, A woman in a kitchen, frame 0.")
    assert refs == [(portrait_bytes, "image/jpeg")]


def test_shot_still_cost_is_measured_and_counted_in_the_total(tmp_path):
    pipeline, _, _ = _pipeline(tmp_path)

    produced = pipeline.produce(_screenplay())

    # 16x16 portrait + 16x16 still, each rounded up to 1MP, per line.
    assert produced.shot_image_cost_usd == pytest.approx(3 * 2 * PRICE)
    assert produced.total_cost_usd == pytest.approx(
        sum(c.cost_usd for c in produced.clips) + produced.portrait_cost_usd + 3 * 2 * PRICE
    )


def test_every_shot_of_a_character_reuses_that_characters_portrait_seed(tmp_path):
    image_client = MockReplicateImageClient(scripted_bytes=_jpeg(tmp_path / "portrait.jpg"))
    pipeline, _, shot_client = _pipeline(tmp_path, shot_types=("wide", "close_up", "medium", "close_up"), image_client=image_client)

    pipeline.produce(_screenplay(4))

    # Lines 0,2 are فوزية and lines 1,3 are جويدة (_screenplay alternates).
    assert shot_client.seeds[0] == shot_client.seeds[2]
    assert shot_client.seeds[1] == shot_client.seeds[3]
    assert shot_client.seeds[0] != shot_client.seeds[1]
    assert sorted(image_client.seeds) == sorted({shot_client.seeds[0], shot_client.seeds[1]})
    assert all(0 <= s < 2**31 for s in shot_client.seeds)


def test_a_two_shot_also_sends_the_scene_partners_portrait(tmp_path):
    pipeline, _, shot_client = _pipeline(tmp_path, shot_types=("two_shot", "close_up", "medium"))

    pipeline.produce(_screenplay())

    assert len(shot_client.calls[0][1]) == 2
    assert len(shot_client.calls[1][1]) == 1


def test_a_director_failure_stops_the_run_before_any_money_is_spent(tmp_path):
    pipeline, video, shot_client = _pipeline(tmp_path, director_reply="not json")

    with pytest.raises(DialoguePipelineError, match="لم يُصرف أي مال حقيقي"):
        pipeline.produce(_screenplay())
    assert video.calls == [] and shot_client.calls == []
    assert pipeline._image_client.calls == []


def test_a_failed_shot_still_fails_the_line_without_calling_the_video_model(tmp_path):
    pipeline, video, _ = _pipeline(tmp_path, shot_client=MockShotImageClient(fail=True))

    produced = pipeline.produce(_screenplay())

    assert video.calls == []
    assert len(produced.failed_lines) == 3
    # The still itself failed, so no "money was spent" note is added.
    assert all("تم صرف مال حقيقي" not in reason for _, reason in produced.failed_lines)
    assert produced.shot_image_cost_usd == 0.0


def test_a_video_failure_after_a_paid_still_says_the_still_was_paid(tmp_path):
    video = MockReplicateVideoClient(scripted_bytes=_mp4(tmp_path / "src.mp4"), fail_at_calls=frozenset({0}))
    pipeline, _, _ = _pipeline(tmp_path, video=video)

    produced = pipeline.produce(_screenplay())

    assert "صورة اللقطة (flux-2-dev)" in produced.failed_lines[0][1]
    assert produced.shot_image_cost_usd == pytest.approx(3 * 2 * PRICE)


def test_cinematic_mode_films_each_line_as_its_own_shot_even_with_multitalk(tmp_path):
    multitalk = MockMultiTalkVideoClient()
    pipeline, video, _ = _pipeline(tmp_path, multitalk_client=multitalk)

    pipeline.produce(_screenplay())

    assert multitalk.calls == []
    assert len(video.calls) == 3


def test_without_the_three_shot_parts_production_is_unchanged(tmp_path):
    video = MockReplicateVideoClient(scripted_bytes=_mp4(tmp_path / "src.mp4"))
    pipeline = DialoguePipeline(
        video, MockTTSClient(), str(tmp_path / "out"),
        image_client=MockReplicateImageClient(scripted_bytes=_jpeg(tmp_path / "portrait.jpg")),
        shot_director=ShotDirectorAgent(MockAgentReasoningClient([])),
    )

    produced = pipeline.produce(_screenplay(1))

    assert not pipeline.cinematic_shots_enabled
    assert video.calls[0]["prompt"] == "stirs the pot" + CINEMATIC_STYLE_SUFFIX
    assert produced.shot_image_cost_usd == 0.0


def test_the_portrait_no_longer_asks_for_eyes_on_the_camera():
    assert "looking directly at camera" not in PORTRAIT_STYLE_SUFFIX


# --- flux-2-dev client and pricing ---

def test_flux2_client_sends_the_documented_input_fields(monkeypatch):
    import httpx

    posted = {}

    class _Resp:
        def __init__(self, status_code=200, data=None, content=b""):
            self.status_code, self._data, self.content, self.text = status_code, data or {}, content, ""

        def json(self):
            return self._data

        def raise_for_status(self):
            pass

    def fake_post(url, *, headers, json, timeout):
        posted["url"], posted["body"] = url, json
        return _Resp(201, {"urls": {"get": "https://api.replicate.com/p/1"}})

    def fake_get(url, *, headers, timeout):
        if url.endswith("/p/1"):
            return _Resp(200, {"status": "succeeded", "output": "https://replicate.delivery/x.jpg"})
        return _Resp(200, content=b"jpeg-bytes")

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    out = RealFlux2ShotImageClient("r8").generate_shot_image("a shot", [(b"face", "image/jpeg")], seed=42)

    assert out == b"jpeg-bytes"
    assert posted["url"] == FLUX2_DEV_ENDPOINT
    assert posted["body"] == {"input": {
        "prompt": "a shot",
        "input_images": ["data:image/jpeg;base64," + base64.b64encode(b"face").decode()],
        "aspect_ratio": "16:9",
        "output_format": "jpg",
        "seed": 42,
    }}


@pytest.mark.parametrize("refs", [[], [(b"x", "image/jpeg")] * 6])
def test_flux2_client_rejects_no_reference_or_more_than_five(refs):
    with pytest.raises(ReplicateImageError):
        RealFlux2ShotImageClient("r8").generate_shot_image("a shot", refs)


def test_flux2_cost_rounds_every_image_up_to_whole_megapixels():
    assert flux2_dev_cost_usd([256], 0.01) == pytest.approx(0.01)
    assert flux2_dev_cost_usd([1_000_000, 1_000_001], 0.01) == pytest.approx(0.03)
    assert estimated_shot_image_usd(0.01) == pytest.approx(0.06)


# --- wiring ---

def _settings(**overrides) -> Settings:
    base = dict(
        groq_api_key="g", groq_model="m", gemini_api_key="gem", gemini_model="m",
        youtube_client_id="", youtube_client_secret="", youtube_refresh_token="",
        telegram_bot_token="tg", azure_speech_key="azure-key", azure_speech_region="eastus",
        azure_tts_lang="ar-SY", replicate_api_token="r8_real",
        runpod_api_key="rp", runpod_s2v_pod_id="pod", runpod_s2v_hourly_rate_usd=0.59,
        multitalk_enabled=False, output_dir="./output-test", max_episode_cost_usd=10.0,
    )
    base.update(overrides)
    return Settings(**base)


def test_build_app_wires_the_cinematic_chain_only_when_the_owner_turns_it_on():
    off = build_app(_settings())
    assert not off.dialogue_pipeline.cinematic_shots_enabled
    assert off.video_client._width == 640

    on = build_app(_settings(cinematic_shots_enabled=True))
    pipeline = on.dialogue_pipeline
    assert pipeline.cinematic_shots_enabled
    assert isinstance(pipeline._shot_image_client, RealFlux2ShotImageClient)
    assert isinstance(pipeline._shot_director._reasoning_client, RealGeminiReasoningClient)
    assert (on.video_client._width, on.video_client._height) == (CINEMATIC_VIDEO_WIDTH, CINEMATIC_VIDEO_HEIGHT)
    assert on.conversation._estimated_shot_image_usd == pytest.approx(estimated_shot_image_usd())


def test_the_switch_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("CINEMATIC_SHOTS_ENABLED", "true")
    assert Settings.from_env().cinematic_shots_enabled
    monkeypatch.setenv("CINEMATIC_SHOTS_ENABLED", "")
    assert not Settings.from_env().cinematic_shots_enabled


def test_gemini_falls_back_to_the_other_provider_when_it_fails(monkeypatch):
    fallback = MockAgentReasoningClient(["from groq"])
    client = RealGeminiReasoningClient("key", "m", fallback=fallback)

    def boom(system_prompt, user_prompt):
        raise RuntimeError("gemini 500")

    monkeypatch.setattr(client, "_complete", boom)
    assert client.complete("s", "u") == "from groq"


def test_the_pre_approve_estimate_adds_one_shot_still_per_spoken_line():
    from src.agents.dialogue_writer import DialogueWriterAgent
    from src.agents.publisher import PublishingAgent
    from src.bot.conversation import BotConversation

    def estimate(shot_usd: float) -> str:
        conv = BotConversation(
            DialogueWriterAgent(MockAgentReasoningClient([])), PublishingAgent({}),
            price_per_second_usd=0.0, price_per_image_usd=0.0, estimated_shot_image_usd=shot_usd,
        )
        return conv._format_screenplay(_screenplay(3))

    assert "$0.00" in estimate(0.0)
    assert "$3.00" in estimate(1.0)  # 3 spoken lines x $1.00


def test_a_grieving_scene_is_read_slower_and_lower_and_a_calm_one_is_not(tmp_path):
    from src.integrations.tts_client import GRIEF_PROSODY
    from src.workflows.dialogue_pipeline import _line_prosody

    screenplay = _screenplay(2)
    screenplay.scenes = [DialogueSceneInfo(index=0, location="hospital room", mood="devastated, quietly heartbroken")]
    assert _line_prosody(screenplay, screenplay.lines[0]) == GRIEF_PROSODY

    screenplay.scenes = [DialogueSceneInfo(index=0, location="old kitchen", mood="warm")]
    screenplay.lines[1].action = "weeping, hands over her face"
    assert _line_prosody(screenplay, screenplay.lines[0]) is None
    assert _line_prosody(screenplay, screenplay.lines[1]) == GRIEF_PROSODY


def test_production_sends_the_grief_prosody_to_azure_and_the_crying_face_to_both_models(tmp_path):
    from src.integrations.tts_client import GRIEF_PROSODY
    from src.workflows.shot_prompt_compiler import GRIEF_FACE_CLAUSE

    tts = MockTTSClient()
    pipeline, video, shot_client = _pipeline(tmp_path, shot_types=("wide", "close_up"))
    pipeline._tts = tts
    screenplay = _screenplay(2)
    screenplay.lines[1].action = "weeping, hands over her face"

    produced = pipeline.produce(screenplay)

    assert len(produced.clips) == 2
    assert tts.prosodies == [None, GRIEF_PROSODY]
    assert GRIEF_FACE_CLAUSE not in shot_client.calls[0][0] and GRIEF_FACE_CLAUSE in shot_client.calls[1][0]
    assert GRIEF_FACE_CLAUSE not in video.calls[0]["prompt"] and GRIEF_FACE_CLAUSE in video.calls[1]["prompt"]
