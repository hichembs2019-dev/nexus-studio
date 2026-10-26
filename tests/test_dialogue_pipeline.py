from __future__ import annotations

import os
import subprocess

import pytest

import base64

from src.integrations.replicate_image_client import (
    PRICE_PER_IMAGE_USD_UNVERIFIED,
    MockReplicateImageClient,
    ReplicateImageError,
)
from src.integrations.multitalk_video_client import MockMultiTalkVideoClient
from src.integrations.replicate_video_client import PROVISIONAL_PRICE_PER_SECOND_USD, MockReplicateVideoClient
from src.integrations.tts_client import MockTTSClient
from src.models import Character, DialogueLine, DialogueScreenplay
from src.workflows.dialogue_pipeline import (
    CAPTION_FONT_PATH,
    CINEMATIC_STYLE_SUFFIX,
    PORTRAIT_STYLE_SUFFIX,
    DialoguePipeline,
    DialoguePipelineError,
)

_ARABIC_FONT_AVAILABLE = os.path.isfile(CAPTION_FONT_PATH)


def _silent_mp4(path: str, duration_seconds: float) -> bytes:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=16:d={duration_seconds}",
         "-pix_fmt", "yuv420p", path],
        capture_output=True, check=True,
    )
    with open(path, "rb") as f:
        return f.read()


def _real_mp3_bytes(path: str, duration_seconds: float = 1.0) -> bytes:
    # The video-probe-fails-but-audio-probe-succeeds recovery path reads
    # the real MP3 ffprobe needs to measure - MockTTSClient's own default
    # scripted bytes (b"fake-mp3-bytes") are plain text, not real audio,
    # which ffprobe correctly refuses to read.
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_seconds}", path],
        capture_output=True, check=True,
    )
    with open(path, "rb") as f:
        return f.read()


def _real_jpeg_bytes(path: str) -> bytes:
    # The caption group photo is built with a real ffmpeg call
    # (_build_caption_group_image) over whatever bytes a generated
    # portrait came back as - MockReplicateImageClient's own default scripted
    # bytes are plain text, not a real image, which ffmpeg correctly
    # refuses to read. Any test that exercises the caption path for
    # real needs an image client that hands back genuine JPEG bytes.
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=16x16", "-frames:v", "1", path],
        capture_output=True, check=True,
    )
    with open(path, "rb") as f:
        return f.read()


# Owner's standing instruction, 2026-10-21 ("احذف الستة صور و قلت التوليد
# حسب القصة" - delete the six photos, I told you generation must be
# per-story): every test below that needs a character to actually have a
# usable photo now goes through the SAME real path production itself
# uses - a MockReplicateImageClient + a real `appearance` string - never a
# pre-made file passed in as a fallback. There is no other way to get a
# character a photo any more.


def test_produce_gives_different_arabic_named_characters_their_own_cached_photo(tmp_path):
    # Real bug found by code review, 2026-10-21: _slug() (for the
    # screenplay title) and the inline char_slug regex (for a
    # character's name) both matched only [a-z0-9] - ASCII. Every real
    # title and character name in this project is Arabic, which has no
    # characters in that class, so both collapsed to the empty string
    # and fell back to the literal default ("episode"/"character")
    # regardless of what the real title/name actually was. The photo
    # cache path is {output_dir}/characters/{slug}_{char_slug}.jpg,
    # which starts with a plain "if the file already exists, reuse it"
    # check - a real cache HIT, not a failure, so nothing is ever
    # reported. The second (and every later) Arabic-named character
    # across every story in the same output_dir would silently reuse
    # the FIRST one's cached photo, at zero cost and with no error at
    # all - exactly the "one shared face" defect the owner has been
    # fighting to eliminate all session, reintroduced by this bug in
    # the very mechanism meant to prevent it.
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    image_client = MockReplicateImageClient()
    pipeline = DialoguePipeline(fal_video, MockTTSClient(), str(tmp_path), image_client=image_client)
    screenplay = DialogueScreenplay(
        title="جسد بلا روح",
        premise="p",
        characters=[
            Character(name="فوزية", appearance="curly dark hair, slender build"),
            Character(name="أحمد", appearance="short grey hair, broad shoulders"),
        ],
        lines=[
            DialogueLine(index=0, character="فوزية", line="واحد", action="a1"),
            DialogueLine(index=1, character="أحمد", line="اثنان", action="a2"),
        ],
    )

    produced = pipeline.produce(screenplay)

    assert len(image_client.calls) == 2  # each character's own real generation was actually attempted
    assert image_client.calls[0] != image_client.calls[1]  # different appearance -> different real prompt
    cached_files = sorted(os.listdir(tmp_path / "characters"))
    assert len(cached_files) == 2, f"expected two distinct cached character photos, got {cached_files}"
    assert len(produced.clips) == 2
    assert produced.portrait_failures == []


def test_produce_fails_every_line_when_no_image_client_is_configured(tmp_path):
    # No fallback face exists at all: without a real image client, NO
    # character can ever get a photo, so every one of their lines fails
    # with zero cost rather than ever using a stand-in.
    fal_video = MockReplicateVideoClient()
    pipeline = DialoguePipeline(fal_video, MockTTSClient(), str(tmp_path))  # no image_client
    screenplay = DialogueScreenplay(
        title="Dinner", premise="p",
        characters=[Character(name="A", appearance="curly dark hair")],
        lines=[DialogueLine(index=0, character="A", line="one", action="a1")],
    )

    produced = pipeline.produce(screenplay)

    assert produced.portrait_cost_usd == 0.0
    assert produced.portrait_failures == [("A", "لا يوجد اتصال حقيقي بمولّد الصور (REPLICATE_API_TOKEN غير مُهيَّأ)")]
    assert len(produced.clips) == 0
    assert len(fal_video.calls) == 0  # never even reached fal.ai - no photo to send it
    assert len(produced.failed_lines) == 1
    assert produced.total_cost_usd == 0.0


def test_produce_fails_a_characters_lines_when_it_has_no_appearance_description(tmp_path):
    image_client = MockReplicateImageClient()
    fal_video = MockReplicateVideoClient()
    pipeline = DialoguePipeline(fal_video, MockTTSClient(), str(tmp_path), image_client=image_client)
    screenplay = DialogueScreenplay(
        title="Dinner", premise="p",
        characters=[Character(name="A")],  # appearance left empty
        lines=[DialogueLine(index=0, character="A", line="one", action="a1")],
    )

    produced = pipeline.produce(screenplay)

    assert image_client.calls == []  # never even asked to generate - nothing to generate from
    assert produced.portrait_failures == [("A", "لا يوجد وصف لهذه الشخصية لتوليد صورة حقيقية منه")]
    assert len(produced.clips) == 0
    assert len(fal_video.calls) == 0  # never reached Replicate at all - no real photo existed to send
    assert len(produced.failed_lines) == 1


def test_produce_fails_the_characters_lines_when_real_portrait_generation_fails(tmp_path):
    class _FailingImageClient:
        def generate_image(self, prompt, seed=None):
            raise ReplicateImageError("Replicate image job failed: simulated real failure")

    fal_video = MockReplicateVideoClient()
    pipeline = DialoguePipeline(fal_video, MockTTSClient(), str(tmp_path), image_client=_FailingImageClient())
    screenplay = DialogueScreenplay(
        title="Dinner", premise="p",
        characters=[Character(name="A", appearance="curly dark hair")],
        lines=[DialogueLine(index=0, character="A", line="one", action="a1")],
    )

    produced = pipeline.produce(screenplay)

    assert len(produced.clips) == 0  # the line never ran - no fallback face, by the owner's order
    assert len(fal_video.calls) == 0  # Replicate video was never called - no photo existed to send
    assert len(produced.failed_lines) == 1
    failed_line, message = produced.failed_lines[0]
    assert failed_line.index == 0
    assert "simulated real failure" in message
    assert produced.portrait_failures == [("A", "Replicate image job failed: simulated real failure")]
    assert produced.portrait_cost_usd == 0.0
    assert produced.total_cost_usd == 0.0


def test_produce_single_line_calls_tts_then_fal_with_data_uris_and_saves_the_clip(tmp_path):
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient(scripted_bytes=b"\xff\xfbfake-mp3")
    image_client = MockReplicateImageClient(scripted_bytes=b"real-generated-portrait-bytes")
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=image_client)
    screenplay = DialogueScreenplay(
        title="Dinner",
        premise="p",
        characters=[Character(name="فوزية", voice="ar-TN-HediNeural", appearance="curly dark hair")],
        lines=[DialogueLine(index=0, character="فوزية", line="العشاء جاهز", action="smiling, setting the table")],
    )

    produced = pipeline.produce(screenplay)

    assert tts.calls == [("العشاء جاهز", "ar-TN-HediNeural", "ar-SY")]
    assert len(fal_video.calls) == 1
    call = fal_video.calls[0]
    assert call["prompt"] == "smiling, setting the table" + CINEMATIC_STYLE_SUFFIX
    assert call["image_data_uri"].startswith("data:image/")
    # The real photo sent is the one actually generated for this story -
    # never a pre-made file.
    sent_image_b64 = call["image_data_uri"].split(",", 1)[1]
    assert base64.b64decode(sent_image_b64) == b"real-generated-portrait-bytes"
    assert call["audio_data_uri"].startswith("data:audio/mpeg;base64,")
    assert call["resolution"] == "480p"

    assert len(produced.clips) == 1
    assert os.path.exists(produced.clips[0].local_path)
    assert produced.stitched_path == produced.clips[0].local_path
    # Real, owner-run test, 2026-10-21: one real Replicate call billed
    # $0.10 for 2s of output - see PROVISIONAL_PRICE_PER_SECOND_USD's own
    # docstring and docs/SOURCES.md.
    assert produced.clips[0].cost_usd == pytest.approx(1.0 * PROVISIONAL_PRICE_PER_SECOND_USD)


def test_produce_uses_the_real_price_per_second_explicitly_passed_in(tmp_path):
    # Real gap found by code review, 2026-10-21: DialoguePipeline must
    # never hard-code a provider's own price constant directly - the
    # real rate for whichever client is actually wired in must be
    # honored explicitly, passed in by the caller (src/app.py). 0.05
    # here is distinct from PROVISIONAL_PRICE_PER_SECOND_USD so this
    # can't pass by coincidence.
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    pipeline = DialoguePipeline(
        fal_video, tts, str(tmp_path), image_client=MockReplicateImageClient(), price_per_second_usd=0.05
    )
    screenplay = DialogueScreenplay(
        title="T", premise="p",
        characters=[Character(name="A", appearance="a")],
        lines=[DialogueLine(index=0, character="A", line="one", action="a1")],
    )

    produced = pipeline.produce(screenplay)

    assert produced.clips[0].cost_usd == pytest.approx(1.0 * 0.05)


def test_produce_uses_the_real_price_per_image_explicitly_passed_in(tmp_path):
    # Owner's cleanup instruction, 2026-10-21 ("قم بالتنظيف ... كثرة
    # الكودات اصبحت تتداخل في بعضها"): found by that same review - the
    # portrait cost had the IDENTICAL defect as price_per_second_usd
    # above (hard-coded to fal.ai's PRICE_PER_IMAGE_USD_UNVERIFIED), just never
    # caught because both placeholder values happened to equal 0.025.
    # 0.09 here is deliberately distinct from every real constant in the
    # codebase so this can't pass by coincidence.
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    pipeline = DialoguePipeline(
        fal_video, MockTTSClient(), str(tmp_path), image_client=MockReplicateImageClient(), price_per_image_usd=0.09
    )
    screenplay = DialogueScreenplay(
        title="T", premise="p",
        characters=[Character(name="A", appearance="a")],
        lines=[DialogueLine(index=0, character="A", line="one", action="a1")],
    )

    produced = pipeline.produce(screenplay)

    assert produced.portrait_cost_usd == pytest.approx(0.09)


def test_produce_falls_back_to_a_default_prompt_when_action_is_empty(tmp_path):
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=MockReplicateImageClient())
    screenplay = DialogueScreenplay(
        title="T",
        premise="p",
        characters=[Character(name="Jouida", appearance="ordinary")],
        lines=[DialogueLine(index=0, character="Jouida", line="hi", action="")],
    )

    pipeline.produce(screenplay)

    assert fal_video.calls[0]["prompt"] == "Jouida speaking" + CINEMATIC_STYLE_SUFFIX


def test_produce_multi_line_stitches_clips_in_order(tmp_path):
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=MockReplicateImageClient())
    screenplay = DialogueScreenplay(
        title="Dinner",
        premise="p",
        characters=[Character(name="A", appearance="a"), Character(name="B", appearance="b")],
        lines=[
            DialogueLine(index=0, character="A", line="one", action="a1"),
            DialogueLine(index=1, character="B", line="two", action="b1"),
        ],
    )

    produced = pipeline.produce(screenplay)

    assert len(produced.clips) == 2
    assert produced.stitched_path is not None
    assert os.path.exists(produced.stitched_path)
    # Two distinct speaking characters -> two real clips AND two real
    # per-story portraits, each billed separately.
    assert produced.total_cost_usd == pytest.approx(2 * 1.0 * PROVISIONAL_PRICE_PER_SECOND_USD + 2 * PRICE_PER_IMAGE_USD_UNVERIFIED)


def test_produce_keeps_real_paid_clips_when_the_final_merge_fails(tmp_path, monkeypatch):
    # Real incident, 2026-10-20: a real run paid for and finished 40.91s
    # of real clips ($8.18, confirmed from fal.ai's own Usage page), then
    # this merge step raised - and because it used to sit outside every
    # try/except, the exception escaped produce() entirely and discarded
    # every one of those real, already-paid, already-on-disk clips. A test
    # that only checks produce() doesn't crash would still pass with the
    # clips thrown away - it must assert the real clips and the real
    # error both survive.
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=MockReplicateImageClient())
    screenplay = DialogueScreenplay(
        title="Dinner",
        premise="p",
        characters=[Character(name="A", appearance="a"), Character(name="B", appearance="b")],
        lines=[
            DialogueLine(index=0, character="A", line="one", action="a1"),
            DialogueLine(index=1, character="B", line="two", action="b1"),
        ],
    )

    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg" and "concat" in cmd:
            class _FailedProcess:
                returncode = 1
                stderr = "simulated real ffmpeg concat failure"
                stdout = ""
            return _FailedProcess()
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    produced = pipeline.produce(screenplay)

    assert len(produced.clips) == 2
    assert all(os.path.exists(c.local_path) for c in produced.clips)
    assert produced.stitched_path is None
    assert produced.stitch_error is not None
    assert "simulated real ffmpeg concat failure" in produced.stitch_error
    assert produced.total_cost_usd == pytest.approx(2 * 1.0 * PROVISIONAL_PRICE_PER_SECOND_USD + 2 * PRICE_PER_IMAGE_USD_UNVERIFIED)


def test_produce_stops_spending_once_the_hard_cost_ceiling_is_reached(tmp_path):
    # Owner's instruction, 2026-10-20 ("قلت لك ضع له كود يشتغل و يسحب
    # المال الا عند صنع فيديو" - put code in that only draws money when a
    # video is actually made): max_cost_usd is a hard ceiling on total
    # real spend within one run. A test that only checks the ceiling
    # exists as a number (never compared against anything) would pass
    # with the guard never actually stopping a real fal.ai call - this
    # asserts the THIRD line's real call never happens at all.
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    # Each clip costs 1.0s * PROVISIONAL_PRICE_PER_SECOND_USD (the real,
    # evidenced rate fal.ai actually bills - see docs/SOURCES.md) = $0.20.
    # The ceiling is only ever checked BEFORE a line is attempted (its
    # real cost isn't known until after fal.ai actually runs it), so a
    # ceiling below one line's cost is what proves the check fires at
    # all: the first line still goes through (nothing spent yet to
    # compare, beyond the one real portrait), every line after it must not.
    pipeline = DialoguePipeline(
        fal_video, tts, str(tmp_path), image_client=MockReplicateImageClient(), max_cost_usd=0.05
    )
    screenplay = DialogueScreenplay(
        title="Dinner",
        premise="p",
        characters=[Character(name="A", appearance="a")],
        lines=[
            DialogueLine(index=0, character="A", line="one", action="a1"),
            DialogueLine(index=1, character="A", line="two", action="a2"),
            DialogueLine(index=2, character="A", line="three", action="a3"),
        ],
    )

    produced = pipeline.produce(screenplay)

    assert len(fal_video.calls) == 1
    assert len(produced.clips) == 1
    assert [l.index for l in produced.skipped_lines] == [1, 2]
    assert produced.total_cost_usd == pytest.approx(PROVISIONAL_PRICE_PER_SECOND_USD + PRICE_PER_IMAGE_USD_UNVERIFIED)


@pytest.mark.skipif(not _ARABIC_FONT_AVAILABLE, reason="fonts-noto-core not installed on this machine - Dockerfile installs it for the real deployment")
def test_produce_still_renders_a_free_caption_line_after_the_cost_ceiling_is_reached(tmp_path):
    # Real bug found by code review, 2026-10-21: the cost ceiling check
    # used to run BEFORE the spoken/caption branch, so a genuinely free
    # ($0, no TTS, no fal.ai call) caption line got skipped into
    # skipped_lines whenever an earlier SPOKEN line had already
    # exhausted the ceiling - even though skipping it saves no real
    # money at all, which contradicts the ceiling's own documented
    # purpose (a ceiling on real SPEND). Reproduced directly: a ceiling
    # that lets exactly one $0.20 spoken line through must still let a
    # LATER $0 caption line render, not be silently dropped with it.
    portrait_bytes = _real_jpeg_bytes(str(tmp_path / "portrait.jpg"))
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    pipeline = DialoguePipeline(
        fal_video, MockTTSClient(), str(tmp_path),
        image_client=MockReplicateImageClient(scripted_bytes=portrait_bytes), max_cost_usd=0.05,
    )
    screenplay = DialogueScreenplay(
        title="Dinner", premise="p",
        characters=[Character(name="A", appearance="a")],
        lines=[
            DialogueLine(index=0, character="A", line="one", action="a1"),  # spoken, $0.20 - exceeds the $0.05 ceiling
            DialogueLine(index=1, character="A", line="نص مجاني", action="", spoken=False),  # $0 - must NOT be skipped
        ],
    )

    produced = pipeline.produce(screenplay)

    assert produced.skipped_lines == []  # the free caption must never be treated as unaffordable
    caption_clip = next((c for c in produced.clips if c.line.index == 1), None)
    assert caption_clip is not None
    assert caption_clip.cost_usd == 0.0


def test_produce_generates_and_uses_a_real_portrait_for_a_character_with_appearance(tmp_path):
    # Owner's instruction, 2026-10-20, rejecting a fixed panel of six
    # generic photos reused across every story forever ("لا تكون صور
    # ثابتة الى الابد ... بل حسب القصة و السيناريو"): a character with a
    # real appearance description must get a real, generated photo used
    # in the actual fal.ai video call.
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    image_client = MockReplicateImageClient(scripted_bytes=b"real-generated-portrait-bytes")
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=image_client)
    screenplay = DialogueScreenplay(
        title="Dinner",
        premise="p",
        characters=[Character(name="A", appearance="curly dark hair, slender build")],
        lines=[DialogueLine(index=0, character="A", line="one", action="a1")],
    )

    produced = pipeline.produce(screenplay)

    assert image_client.calls == [
        "photorealistic portrait photo of a person, curly dark hair, slender build, in a scene from this real story: p"
        + PORTRAIT_STYLE_SUFFIX
    ]
    sent_image_b64 = fal_video.calls[0]["image_data_uri"].split(",", 1)[1]
    assert base64.b64decode(sent_image_b64) == b"real-generated-portrait-bytes"
    assert produced.portrait_cost_usd == pytest.approx(PRICE_PER_IMAGE_USD_UNVERIFIED)
    assert produced.total_cost_usd == pytest.approx(PROVISIONAL_PRICE_PER_SECOND_USD + PRICE_PER_IMAGE_USD_UNVERIFIED)


def test_produce_sends_the_characters_real_gender_into_the_portrait_prompt(tmp_path):
    # Real bug, photographed 2026-10-22: a character written and
    # spoken-of as "الشاب" (the young man, gender="male" from the cast
    # step) got a real woman's face in production - the portrait prompt
    # had no gender signal at all, only appearance's free-text sentence,
    # which does not reliably name a sex. Character.gender must now
    # reach the real prompt, for both real genders, not just "default".
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    image_client = MockReplicateImageClient()
    pipeline = DialoguePipeline(fal_video, MockTTSClient(), str(tmp_path), image_client=image_client)
    screenplay = DialogueScreenplay(
        title="T", premise="p",
        characters=[
            Character(name="A", appearance="short hair, broad build", gender="male"),
            Character(name="B", appearance="long hair, slender build", gender="female"),
        ],
        lines=[
            DialogueLine(index=0, character="A", line="one", action="a1"),
            DialogueLine(index=1, character="B", line="two", action="a2"),
        ],
    )

    pipeline.produce(screenplay)

    assert image_client.calls[0].startswith("photorealistic portrait photo of a man, short hair, broad build")
    assert image_client.calls[1].startswith("photorealistic portrait photo of a woman, long hair, slender build")


def test_produce_sends_the_real_premise_as_scene_context_capped_to_300_chars(tmp_path):
    # Real finding, 2026-10-22: the owner compared a real produced clip
    # against the project's first real video (two people in a real
    # kitchen) and named the real difference precisely - "الفرق بينهم في
    # المشهد" (the difference between them is in the scene). Every
    # portrait used to be an isolated studio headshot with a plain,
    # empty background, regardless of how real or balanced the dialogue
    # was. The real story's own premise must now reach the image prompt
    # so the model has something to place the character INSIDE.
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    long_premise = "مطبخ دافئ في بيت العائلة، " * 20  # well over 300 real chars

    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    image_client = MockReplicateImageClient()
    pipeline = DialoguePipeline(fal_video, MockTTSClient(), str(tmp_path), image_client=image_client)
    screenplay = DialogueScreenplay(
        title="T", premise=long_premise,
        characters=[Character(name="A", appearance="a")],
        lines=[DialogueLine(index=0, character="A", line="one", action="a1")],
    )

    pipeline.produce(screenplay)

    sent_prompt = image_client.calls[0]
    assert "in a scene from this real story:" in sent_prompt
    assert long_premise[:300] in sent_prompt
    assert long_premise[:301] not in sent_prompt  # genuinely capped, not the full premise


def test_produce_reuses_the_cached_portrait_without_regenerating_on_a_second_call(tmp_path):
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    image_client = MockReplicateImageClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=image_client)
    screenplay = DialogueScreenplay(
        title="Dinner", premise="p",
        characters=[Character(name="A", appearance="curly dark hair")],
        lines=[DialogueLine(index=0, character="A", line="one", action="a1")],
    )

    pipeline.produce(screenplay)
    second = pipeline.produce(screenplay)

    assert len(image_client.calls) == 1  # not regenerated - cached on disk from the first call
    assert second.portrait_cost_usd == 0.0


def test_save_character_photo_is_picked_up_as_a_cache_hit_with_zero_real_cost(tmp_path):
    # Owner's instruction, 2026-10-21 ("الصور ضعها انت الان و هكذا لا
    # ندفع شيء" - you put the photos in yourself, so we don't pay
    # anything): save_character_photo() writes to the EXACT path
    # _resolve_character_photo() checks first - proven here by producing
    # a real screenplay for this exact character and asserting the
    # image client is never called and the real cost is $0.
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    image_client = MockReplicateImageClient()
    pipeline = DialoguePipeline(fal_video, MockTTSClient(), str(tmp_path), image_client=image_client)
    screenplay = DialogueScreenplay(
        title="Dinner", premise="p",
        characters=[Character(name="فوزية", appearance="curly dark hair")],
        lines=[DialogueLine(index=0, character="فوزية", line="one", action="a1")],
    )

    saved_path = pipeline.save_character_photo("Dinner", "فوزية", b"owner-provided-real-photo-bytes")
    produced = pipeline.produce(screenplay)

    assert os.path.exists(saved_path)
    with open(saved_path, "rb") as f:
        assert f.read() == b"owner-provided-real-photo-bytes"
    assert len(image_client.calls) == 0  # never generated - the owner's own photo was already cached
    assert produced.portrait_cost_usd == 0.0
    sent_image_b64 = fal_video.calls[0]["image_data_uri"].split(",", 1)[1]
    assert base64.b64decode(sent_image_b64) == b"owner-provided-real-photo-bytes"


def test_save_character_photo_overwrites_an_existing_cached_photo(tmp_path):
    pipeline = DialoguePipeline(MockReplicateVideoClient(), MockTTSClient(), str(tmp_path), image_client=MockReplicateImageClient())

    first_path = pipeline.save_character_photo("Dinner", "A", b"old-photo")
    second_path = pipeline.save_character_photo("Dinner", "A", b"new-photo")

    assert first_path == second_path
    with open(second_path, "rb") as f:
        assert f.read() == b"new-photo"


def test_produce_portrait_generation_respects_the_hard_cost_ceiling(tmp_path):
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    image_client = MockReplicateImageClient()
    # The ceiling allows the FIRST portrait through (nothing spent yet to
    # compare against) but must block the second real portrait call -
    # same "checked before, not after" principle as the per-line ceiling.
    pipeline = DialoguePipeline(
        fal_video, MockTTSClient(), str(tmp_path),
        image_client=image_client, max_cost_usd=PRICE_PER_IMAGE_USD_UNVERIFIED / 2,
    )
    screenplay = DialogueScreenplay(
        title="Dinner", premise="p",
        characters=[
            Character(name="A", appearance="curly dark hair"),
            Character(name="B", appearance="short grey hair"),
        ],
        lines=[
            DialogueLine(index=0, character="A", line="one", action="a1"),
            DialogueLine(index=1, character="B", line="two", action="b1"),
        ],
    )

    pipeline.produce(screenplay)

    assert len(image_client.calls) == 1  # the second character's portrait was never attempted


@pytest.mark.skipif(not _ARABIC_FONT_AVAILABLE, reason="fonts-noto-core not installed on this machine - Dockerfile installs it for the real deployment")
def test_produce_renders_a_spoken_false_line_as_a_zero_cost_caption(tmp_path):
    # Owner's instruction, 2026-10-20 ("نحذف الام و يبقى حوارها مكتوب على
    # الشاشة ... لكي نستحقظ على المال" - remove the mother as a speaking
    # role, her dialogue stays written on screen, to save money): a
    # spoken=False line must cost nothing real at all (no TTS, no fal.ai
    # call) and still produce a real playable clip - a test that only
    # checks produce() doesn't crash would pass even if this silently
    # fell back to generating a real (billed) clip anyway.
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    portrait_bytes = _real_jpeg_bytes(str(tmp_path / "portrait.jpg"))

    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=MockReplicateImageClient(scripted_bytes=portrait_bytes))
    screenplay = DialogueScreenplay(
        title="Dinner",
        premise="p",
        characters=[
            Character(name="A", appearance="a"),
            Character(name="B", appearance="b"),
        ],
        lines=[
            DialogueLine(index=0, character="A", line="one", action="a1"),
            DialogueLine(index=1, character="B", line="زوجتك لم تعد تحترمك", action="", spoken=False),
        ],
    )

    produced = pipeline.produce(screenplay)

    assert len(fal_video.calls) == 1  # only the spoken line ever called fal.ai
    assert len(tts.calls) == 1  # only the spoken line ever called TTS
    assert len(produced.clips) == 2
    caption_clip = next(c for c in produced.clips if c.line.index == 1)
    assert caption_clip.cost_usd == 0.0
    assert os.path.exists(caption_clip.local_path)
    import subprocess as _sp
    probe = _sp.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", caption_clip.local_path],
        capture_output=True, text=True,
    )
    assert probe.returncode == 0
    assert float(probe.stdout.strip()) >= 2.0  # CAPTION_MIN_SECONDS floor for a short line
    # The one spoken line's real cost, plus both characters' real portraits
    # (B needs one too, to appear in the caption's shared group photo).
    assert produced.total_cost_usd == pytest.approx(PROVISIONAL_PRICE_PER_SECOND_USD + 2 * PRICE_PER_IMAGE_USD_UNVERIFIED)


@pytest.mark.skipif(not _ARABIC_FONT_AVAILABLE, reason="fonts-noto-core not installed on this machine - Dockerfile installs it for the real deployment")
def test_produce_captions_a_line_even_when_its_own_character_has_no_photo(tmp_path):
    # Documented limitation (README): the caption group photo is built
    # from whichever captioned characters' photos actually resolved - a
    # caption LINE still costs $0 and still renders even if its own
    # character's portrait failed, as long as at least one other
    # captioned character's photo is available to show.
    portrait_bytes = _real_jpeg_bytes(str(tmp_path / "portrait.jpg"))
    fal_video = MockReplicateVideoClient()
    pipeline = DialoguePipeline(fal_video, MockTTSClient(), str(tmp_path), image_client=MockReplicateImageClient(scripted_bytes=portrait_bytes))
    screenplay = DialogueScreenplay(
        title="Dinner", premise="p",
        characters=[
            Character(name="A"),  # no appearance - its own photo will fail
            Character(name="B", appearance="b"),
        ],
        lines=[
            DialogueLine(index=0, character="A", line="نص أ", action="", spoken=False),
            DialogueLine(index=1, character="B", line="نص ب", action="", spoken=False),
        ],
    )

    produced = pipeline.produce(screenplay)

    assert len(produced.clips) == 2  # both captions rendered, using B's photo in the group image
    assert all(c.cost_usd == 0.0 for c in produced.clips)
    assert produced.portrait_failures == [("A", "لا يوجد وصف لهذه الشخصية لتوليد صورة حقيقية منه")]


@pytest.mark.skipif(not _ARABIC_FONT_AVAILABLE, reason="fonts-noto-core not installed on this machine - Dockerfile installs it for the real deployment")
def test_produce_fails_every_caption_when_no_captioned_character_has_a_photo(tmp_path):
    fal_video = MockReplicateVideoClient()
    pipeline = DialoguePipeline(fal_video, MockTTSClient(), str(tmp_path))  # no image_client at all
    screenplay = DialogueScreenplay(
        title="Dinner", premise="p",
        characters=[Character(name="A")],
        lines=[DialogueLine(index=0, character="A", line="نص", action="", spoken=False)],
    )

    produced = pipeline.produce(screenplay)

    assert len(produced.clips) == 0
    assert len(produced.failed_lines) == 1
    assert "لا توجد صورة حقيقية" in produced.failed_lines[0][1]


@pytest.mark.skipif(not _ARABIC_FONT_AVAILABLE, reason="fonts-noto-core not installed on this machine - Dockerfile installs it for the real deployment")
def test_produce_reports_a_real_caption_rendering_failure_without_a_billed_prefix(tmp_path, monkeypatch):
    # A spoken=False line that fails locally never spent real money, so
    # it must never carry the "تم صرف مال حقيقي" prefix the spoken path
    # uses for a billed-then-failed line.
    portrait_bytes = _real_jpeg_bytes(str(tmp_path / "portrait.jpg"))
    pipeline = DialoguePipeline(
        MockReplicateVideoClient(), MockTTSClient(), str(tmp_path),
        image_client=MockReplicateImageClient(scripted_bytes=portrait_bytes),
    )
    screenplay = DialogueScreenplay(
        title="Dinner", premise="p",
        characters=[Character(name="A", appearance="a")],
        lines=[DialogueLine(index=0, character="A", line="نص", action="", spoken=False)],
    )

    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg" and "drawtext" in " ".join(cmd):
            class _Failed:
                returncode = 1
                stderr = "simulated real caption render failure"
                stdout = ""
            return _Failed()
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    produced = pipeline.produce(screenplay)

    assert len(produced.clips) == 0
    assert len(produced.failed_lines) == 1
    assert "تم صرف مال حقيقي" not in produced.failed_lines[0][1]
    assert "simulated real caption render failure" in produced.failed_lines[0][1]


@pytest.mark.skipif(not _ARABIC_FONT_AVAILABLE, reason="fonts-noto-core not installed on this machine - Dockerfile installs it for the real deployment")
def test_produce_does_not_reuse_a_stale_caption_group_photo_after_the_caption_roster_changes(tmp_path):
    # Real bug found by code review, 2026-10-21: the caption group photo
    # cache used to be keyed only by the screenplay's own title-derived
    # slug - a retry of the SAME screenplay (same title, same slug - the
    # exact reason per-character photos are deliberately cached this way,
    # see _resolve_character_photo's own docstring) after an /edit or
    # /caption that changed WHICH characters are captioned would silently
    # reuse the group photo built for the FIRST, now-stale, roster - the
    # wrong faces shown in every caption frame, with no error anywhere.
    portrait_a = _real_jpeg_bytes(str(tmp_path / "a.jpg"))
    pipeline_a = DialoguePipeline(
        MockReplicateVideoClient(), MockTTSClient(), str(tmp_path), image_client=MockReplicateImageClient(scripted_bytes=portrait_a)
    )
    screenplay_a = DialogueScreenplay(
        title="Same Title", premise="p",
        characters=[Character(name="A", appearance="a")],
        lines=[DialogueLine(index=0, character="A", line="نص أ", action="", spoken=False)],
    )
    pipeline_a.produce(screenplay_a)

    portrait_b = _real_jpeg_bytes(str(tmp_path / "b.jpg"))
    pipeline_b = DialoguePipeline(
        MockReplicateVideoClient(), MockTTSClient(), str(tmp_path), image_client=MockReplicateImageClient(scripted_bytes=portrait_b)
    )
    screenplay_b = DialogueScreenplay(
        title="Same Title", premise="p",  # same title -> same slug, like a real retry after /edit
        characters=[Character(name="B", appearance="b")],
        lines=[DialogueLine(index=0, character="B", line="نص ب", action="", spoken=False)],
    )
    pipeline_b.produce(screenplay_b)

    group_images = [f for f in os.listdir(str(tmp_path)) if "_caption_group" in f]
    # Two genuinely different caption rosters under the same screenplay
    # title must produce two distinct group photo files - never one
    # shared (and therefore wrong) file.
    assert len(group_images) == 2


def test_produce_rejects_a_line_whose_speaker_is_not_in_the_cast(tmp_path):
    pipeline = DialoguePipeline(MockReplicateVideoClient(), MockTTSClient(), str(tmp_path))
    screenplay = DialogueScreenplay(
        title="T", premise="p", characters=[],
        lines=[DialogueLine(index=0, character="Ghost", line="boo", action="")],
    )

    with pytest.raises(DialoguePipelineError, match="not in the cast"):
        pipeline.produce(screenplay)


def test_produce_keeps_already_paid_for_clips_when_a_later_line_fails(tmp_path):
    # Real finding, 2026-10-19: a real 22-line run paid for and
    # successfully generated ~15 real clips (fal.ai's own dashboard
    # confirmed zero errors on its side), then one line's real job ran
    # past the client's own 600s wait - and produce() used to let that
    # exception propagate straight out, discarding every clip already
    # paid for and already sitting finished on disk. A test that only
    # checks "produce() raises on a fal.ai failure" would still pass
    # with that defect in place - it has to check the earlier real
    # clips survive.
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes, fail_at_calls=frozenset({1}))
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=MockReplicateImageClient())
    screenplay = DialogueScreenplay(
        title="T",
        premise="p",
        characters=[Character(name="A", appearance="a")],
        lines=[
            DialogueLine(index=0, character="A", line="one", action="a1"),
            DialogueLine(index=1, character="A", line="two", action="a2"),
            DialogueLine(index=2, character="A", line="three", action="a3"),
        ],
    )

    produced = pipeline.produce(screenplay)

    assert len(produced.clips) == 2  # line 0 and line 2 succeeded
    assert [c.line.index for c in produced.clips] == [0, 2]
    assert len(produced.failed_lines) == 1
    failed_line, error_message = produced.failed_lines[0]
    assert failed_line.index == 1
    assert "mock failure at call 1" in error_message
    # the fal.ai call itself raised and was never billed for this line -
    # must NOT carry the "real money spent" note
    assert "تم صرف مال حقيقي" not in error_message
    # the real money already spent on the surviving clips (plus the one
    # real portrait, generated once and reused for all three lines) must
    # still be reported, not thrown away along with the failed one
    assert produced.total_cost_usd == pytest.approx(2 * 1.0 * PROVISIONAL_PRICE_PER_SECOND_USD + PRICE_PER_IMAGE_USD_UNVERIFIED)


def test_produce_reports_when_a_line_was_billed_by_fal_ai_before_the_failure(tmp_path, monkeypatch):
    # Real bug found by code review, 2026-10-19: a single generic
    # failure message used to let the bot claim EVERY failed line's
    # cost "was not counted" - false whenever fal.ai's own real,
    # billed call already succeeded and only a LATER local step (here:
    # ffprobe) failed. A test that only checks a FalVideoError-style
    # message would miss this - it has to check the real billing note
    # actually appears when the paid call itself succeeded.
    #
    # Real bug found by code review, 2026-10-21 ("لا يجب ان يفشل شيء
    # لاني ادفع مالا عند الفشل" - nothing may fail, because I pay money
    # when it fails): when ffprobe ALONE fails on an already-paid,
    # already-downloaded real video, discarding the clip entirely meant
    # the owner paid fal.ai for a video that genuinely exists on the
    # server and never reached them - not even as a file to retrieve.
    # The clip must now be recovered and delivered regardless.
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=MockReplicateImageClient())
    screenplay = DialogueScreenplay(
        title="T", premise="p",
        characters=[Character(name="A", appearance="a")],
        lines=[DialogueLine(index=0, character="A", line="one", action="a1")],
    )

    import src.workflows.dialogue_pipeline as dialogue_pipeline_module

    def _broken_ffprobe(path):
        raise RuntimeError("ffprobe missing")

    monkeypatch.setattr(dialogue_pipeline_module, "_ffprobe_duration", _broken_ffprobe)

    produced = pipeline.produce(screenplay)

    # The real, already-paid video is still delivered - never discarded
    # just because its exact duration could not be measured.
    assert len(produced.clips) == 1
    assert os.path.exists(produced.clips[0].local_path)
    assert produced.clips[0].cost_usd == 0.0  # genuinely unmeasured, never guessed
    assert len(produced.failed_lines) == 1
    _, error_message = produced.failed_lines[0]
    assert "تم صرف مال حقيقي" in error_message
    assert "مرفق أدناه" in error_message  # must tell the owner the real file is still attached
    assert len(fal_video.calls) == 1  # the real, billed call did happen


def test_produce_falls_back_to_the_audio_duration_when_only_the_video_probe_fails(tmp_path, monkeypatch):
    # Owner's standing instruction, 2026-10-21 ("لا يجب ان يفشل شيء
    # لاني ادفع مالا عند الفشل"): the smarter recovery path - when only
    # the VIDEO probe fails (a corrupt/partial response from fal.ai) but
    # the real audio file (written successfully moments earlier, and
    # lip-synced 1:1 to the video) can still be measured, its real,
    # measured duration is used for the real cost - not a guess, not a
    # silent $0.
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    mp3_bytes = _real_mp3_bytes(str(tmp_path / "ref_audio.mp3"), duration_seconds=1.0)

    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient(scripted_bytes=mp3_bytes)
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=MockReplicateImageClient())
    screenplay = DialogueScreenplay(
        title="T", premise="p",
        characters=[Character(name="A", appearance="a")],
        lines=[DialogueLine(index=0, character="A", line="one", action="a1")],
    )

    import src.workflows.dialogue_pipeline as dialogue_pipeline_module

    real_probe = dialogue_pipeline_module._ffprobe_duration

    def _probe_fails_only_for_video(path):
        if path.endswith(".mp4"):
            raise RuntimeError("corrupt video response")
        return real_probe(path)

    monkeypatch.setattr(dialogue_pipeline_module, "_ffprobe_duration", _probe_fails_only_for_video)

    produced = pipeline.produce(screenplay)

    assert len(produced.clips) == 1
    assert produced.clips[0].cost_usd > 0.0  # a real, measured cost - not the $0 last-resort case
    assert len(produced.failed_lines) == 0  # fully recovered, nothing left to report as failed


# Real wiring, 2026-10-23: MultiTalk two-speaker pairing tests. Every
# real behavioral guarantee here exists BECAUSE of the real crash/fix
# history in docs/SOURCES.md - see dialogue_pipeline.py's own comment at
# the real pairing site for why pairing is scoped to exactly two
# CONSECUTIVE lines, same scene, different speakers (never a whole
# multi-turn scene in one MultiTalk call).


def _two_speaker_screenplay(**line_kwargs) -> DialogueScreenplay:
    lines = line_kwargs.get("lines") or [
        DialogueLine(index=0, character="فوزية", line="مرحبا", action="smiling", scene=0),
        DialogueLine(index=1, character="جويدة", line="أهلا", action="waving", scene=0),
    ]
    return DialogueScreenplay(
        title="T", premise="p",
        characters=[
            Character(name="فوزية", appearance="curly dark hair"),
            Character(name="جويدة", appearance="straight brown hair"),
        ],
        lines=lines,
    )


def test_produce_pairs_two_consecutive_different_speakers_into_one_real_multitalk_clip(tmp_path):
    portrait_path = str(tmp_path / "portrait.jpg")
    multitalk_mp4 = _silent_mp4(str(tmp_path / "mt.mp4"), duration_seconds=4.0)

    fal_video = MockReplicateVideoClient()
    tts = MockTTSClient()
    image_client = MockReplicateImageClient(scripted_bytes=_real_jpeg_bytes(portrait_path))
    multitalk = MockMultiTalkVideoClient(scripted_bytes=multitalk_mp4)
    pipeline = DialoguePipeline(
        fal_video, tts, str(tmp_path), image_client=image_client, multitalk_client=multitalk
    )
    screenplay = _two_speaker_screenplay()

    produced = pipeline.produce(screenplay)

    # One real clip covering both lines - NOT two separate single-speaker
    # clips, and the single-speaker fal_video client must never be called
    # for either of these two lines.
    assert len(produced.clips) == 1
    assert len(fal_video.calls) == 0
    assert len(multitalk.calls) == 1
    clip = produced.clips[0]
    assert [l.index for l in clip.covered_lines] == [0, 1]
    assert clip.cost_usd == pytest.approx(4.0 * PROVISIONAL_PRICE_PER_SECOND_USD)
    assert produced.stitched_path == clip.local_path


def test_produce_sends_each_speakers_own_real_audio_separately_to_multitalk(tmp_path):
    portrait_path = str(tmp_path / "portrait.jpg")
    multitalk_mp4 = _silent_mp4(str(tmp_path / "mt.mp4"), duration_seconds=2.0)

    tts = MockTTSClient(scripted_bytes=b"\xff\xfbfake-mp3")
    multitalk = MockMultiTalkVideoClient(scripted_bytes=multitalk_mp4)
    pipeline = DialoguePipeline(
        MockReplicateVideoClient(), tts, str(tmp_path),
        image_client=MockReplicateImageClient(scripted_bytes=_real_jpeg_bytes(portrait_path)),
        multitalk_client=multitalk,
    )
    screenplay = _two_speaker_screenplay()

    pipeline.produce(screenplay)

    assert tts.calls == [("مرحبا", "ar-TN-HediNeural", "ar-SY"), ("أهلا", "ar-TN-HediNeural", "ar-SY")]
    call = multitalk.calls[0]
    assert call["speaker_1_audio_bytes"] == b"\xff\xfbfake-mp3"
    assert call["speaker_2_audio_bytes"] == b"\xff\xfbfake-mp3"


def test_produce_does_not_pair_lines_from_different_scenes(tmp_path):
    # Real correctness requirement, not just an optimization: pairing
    # across a real scene break would put two characters from two
    # different real scenes into one shared-location video.
    portrait_path = str(tmp_path / "portrait.jpg")
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    multitalk = MockMultiTalkVideoClient()
    pipeline = DialoguePipeline(
        fal_video, MockTTSClient(), str(tmp_path),
        image_client=MockReplicateImageClient(scripted_bytes=_real_jpeg_bytes(portrait_path)),
        multitalk_client=multitalk,
    )
    screenplay = _two_speaker_screenplay(
        lines=[
            DialogueLine(index=0, character="فوزية", line="مرحبا", action="a", scene=0),
            DialogueLine(index=1, character="جويدة", line="أهلا", action="b", scene=1),
        ]
    )

    produced = pipeline.produce(screenplay)

    assert len(multitalk.calls) == 0
    assert len(fal_video.calls) == 2
    assert len(produced.clips) == 2


def test_produce_does_not_pair_the_same_speaker_with_themselves(tmp_path):
    portrait_path = str(tmp_path / "portrait.jpg")
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    multitalk = MockMultiTalkVideoClient()
    pipeline = DialoguePipeline(
        fal_video, MockTTSClient(), str(tmp_path),
        image_client=MockReplicateImageClient(scripted_bytes=_real_jpeg_bytes(portrait_path)),
        multitalk_client=multitalk,
    )
    screenplay = _two_speaker_screenplay(
        lines=[
            DialogueLine(index=0, character="فوزية", line="واحد", action="a", scene=0),
            DialogueLine(index=1, character="فوزية", line="اثنان", action="b", scene=0),
        ]
    )

    produced = pipeline.produce(screenplay)

    assert len(multitalk.calls) == 0
    assert len(fal_video.calls) == 2


def test_produce_does_not_pair_a_spoken_line_with_a_following_caption(tmp_path):
    portrait_path = str(tmp_path / "portrait.jpg")
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    multitalk = MockMultiTalkVideoClient()
    pipeline = DialoguePipeline(
        fal_video, MockTTSClient(), str(tmp_path),
        image_client=MockReplicateImageClient(scripted_bytes=_real_jpeg_bytes(portrait_path)),
        multitalk_client=multitalk,
    )
    screenplay = _two_speaker_screenplay(
        lines=[
            DialogueLine(index=0, character="فوزية", line="مرحبا", action="a", scene=0),
            DialogueLine(index=1, character="جويدة", line="أهلا", action="b", scene=0, spoken=False),
        ]
    )

    produced = pipeline.produce(screenplay)

    assert len(multitalk.calls) == 0
    assert len(fal_video.calls) == 1


def test_produce_falls_back_to_single_speaker_when_no_multitalk_client_is_configured(tmp_path):
    # Real backward-compatibility guarantee: multitalk_client defaults to
    # None, so every existing production run (and every existing test)
    # keeps behaving exactly as before.
    portrait_path = str(tmp_path / "portrait.jpg")
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    pipeline = DialoguePipeline(
        fal_video, MockTTSClient(), str(tmp_path),
        image_client=MockReplicateImageClient(scripted_bytes=_real_jpeg_bytes(portrait_path)),
    )
    screenplay = _two_speaker_screenplay()

    produced = pipeline.produce(screenplay)

    assert len(fal_video.calls) == 2
    assert len(produced.clips) == 2


def test_produce_falls_back_when_one_characters_real_photo_is_missing(tmp_path):
    # A pair needs BOTH real photos - if one character's portrait
    # genuinely failed, the pair never forms; the single-speaker path
    # then correctly fails only that one character's line while still
    # attempting the other's.
    portrait_path = str(tmp_path / "portrait.jpg")
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    fal_video = MockReplicateVideoClient(scripted_bytes=clip_bytes)
    multitalk = MockMultiTalkVideoClient()
    pipeline = DialoguePipeline(
        fal_video, MockTTSClient(), str(tmp_path),
        image_client=None,  # no image client at all - no real photo for anyone
        multitalk_client=multitalk,
    )
    screenplay = _two_speaker_screenplay()

    produced = pipeline.produce(screenplay)

    assert len(multitalk.calls) == 0
    assert len(produced.failed_lines) == 2


def test_produce_skips_a_real_pair_together_once_the_cost_ceiling_is_reached(tmp_path):
    # Real gap this test must actually exercise: the ceiling is reached
    # AFTER both real portraits (zero-cost prerequisites for pairing at
    # all) but BEFORE the real video call - a ceiling of exactly 0.0
    # would instead be caught by the earlier portrait-generation ceiling
    # check, never reaching the pair-specific one this test exists for.
    portrait_path = str(tmp_path / "portrait.jpg")
    multitalk_mp4 = _silent_mp4(str(tmp_path / "mt.mp4"), duration_seconds=4.0)
    multitalk = MockMultiTalkVideoClient(scripted_bytes=multitalk_mp4)
    pipeline = DialoguePipeline(
        MockReplicateVideoClient(), MockTTSClient(), str(tmp_path),
        image_client=MockReplicateImageClient(scripted_bytes=_real_jpeg_bytes(portrait_path)),
        multitalk_client=multitalk,
        max_cost_usd=2 * PRICE_PER_IMAGE_USD_UNVERIFIED,  # covers both real portraits, nothing left for video
    )
    screenplay = _two_speaker_screenplay()

    produced = pipeline.produce(screenplay)

    assert produced.portrait_cost_usd == pytest.approx(2 * PRICE_PER_IMAGE_USD_UNVERIFIED)  # both real photos generated
    assert len(multitalk.calls) == 0
    assert [l.index for l in produced.skipped_lines] == [0, 1]
    assert produced.clips == []


def test_produce_fails_both_paired_lines_when_the_real_multitalk_call_fails(tmp_path):
    portrait_path = str(tmp_path / "portrait.jpg")
    multitalk = MockMultiTalkVideoClient(fail_at_calls=frozenset({0}))
    pipeline = DialoguePipeline(
        MockReplicateVideoClient(), MockTTSClient(), str(tmp_path),
        image_client=MockReplicateImageClient(scripted_bytes=_real_jpeg_bytes(portrait_path)),
        multitalk_client=multitalk,
    )
    screenplay = _two_speaker_screenplay()

    produced = pipeline.produce(screenplay)

    assert produced.clips == []
    assert len(produced.failed_lines) == 2
    assert [l.index for l, _ in produced.failed_lines] == [0, 1]
    # the real call raised before any billing happened - must not carry
    # the "money already spent" note
    for _, message in produced.failed_lines:
        assert "تم صرف مال حقيقي" not in message


def test_produce_reports_billed_money_when_only_the_real_multitalk_probe_fails(tmp_path, monkeypatch):
    portrait_path = str(tmp_path / "portrait.jpg")
    multitalk_mp4 = _silent_mp4(str(tmp_path / "mt.mp4"), duration_seconds=3.0)
    multitalk = MockMultiTalkVideoClient(scripted_bytes=multitalk_mp4)
    pipeline = DialoguePipeline(
        MockReplicateVideoClient(), MockTTSClient(), str(tmp_path),
        image_client=MockReplicateImageClient(scripted_bytes=_real_jpeg_bytes(portrait_path)),
        multitalk_client=multitalk,
    )
    screenplay = _two_speaker_screenplay()

    import src.workflows.dialogue_pipeline as dialogue_pipeline_module

    monkeypatch.setattr(
        dialogue_pipeline_module, "_ffprobe_duration",
        lambda path: (_ for _ in ()).throw(RuntimeError("ffprobe missing")),
    )

    produced = pipeline.produce(screenplay)

    assert len(produced.clips) == 1  # the real, already-paid file is still delivered
    assert produced.clips[0].cost_usd == 0.0
    assert len(produced.failed_lines) == 1
    _, message = produced.failed_lines[0]
    assert "تم صرف مال حقيقي" in message


def test_build_two_speaker_reference_caches_the_real_composite_and_masks(tmp_path):
    portrait_1 = str(tmp_path / "p1.jpg")
    portrait_2 = str(tmp_path / "p2.jpg")
    _real_jpeg_bytes(portrait_1)
    _real_jpeg_bytes(portrait_2)
    pipeline = DialoguePipeline(MockReplicateVideoClient(), MockTTSClient(), str(tmp_path))

    ref_path, mask_1, mask_2 = pipeline._build_two_speaker_reference(portrait_1, portrait_2, "ep", "a", "b")

    assert os.path.isfile(ref_path)
    assert os.path.isfile(mask_1)
    assert os.path.isfile(mask_2)
    mtime_before = os.path.getmtime(ref_path)

    # second call for the exact same pair must be a real cache hit - not rebuilt
    ref_path_2, _, _ = pipeline._build_two_speaker_reference(portrait_1, portrait_2, "ep", "a", "b")
    assert ref_path_2 == ref_path
    assert os.path.getmtime(ref_path) == mtime_before


class _ReportingVideoClient(MockReplicateVideoClient):
    """A provider that reports each clip's own real cost (Kling) and can
    fail AFTER part of the clip was paid for."""

    def __init__(self, scripted_bytes: bytes, costs: list[float | None], errors: list[Exception | None]):
        super().__init__(scripted_bytes=scripted_bytes)
        self._costs = costs
        self._errors = errors
        self._cost: float | None = None

    def last_cost_usd(self):
        return self._cost

    def generate_speech_to_video(self, **kwargs):
        index = len(self.calls)
        super().generate_speech_to_video(**kwargs)
        self._cost = self._costs[index]
        if self._errors[index] is not None:
            raise self._errors[index]
        return self._scripted_bytes


def _two_line_screenplay() -> DialogueScreenplay:
    return DialogueScreenplay(
        title="t", premise="p",
        characters=[Character(name="A", appearance="short dark hair")],
        lines=[
            DialogueLine(index=0, character="A", line="واحد", action="a1"),
            DialogueLine(index=1, character="A", line="اثنان", action="a2"),
        ],
    )


def test_produce_uses_the_providers_own_reported_clip_cost_and_counts_partial_spend(tmp_path):
    from src.integrations.kling_video_client import KlingVideoError

    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    video = _ReportingVideoClient(clip_bytes, costs=[0.355, 0.43], errors=[None, KlingVideoError("no face")])
    pipeline = DialoguePipeline(
        video, MockTTSClient(), str(tmp_path), image_client=MockReplicateImageClient(), price_per_second_usd=99.0
    )

    produced = pipeline.produce(_two_line_screenplay())

    # Reported cost wins over seconds x rate (which would be ~$99).
    assert produced.clips[0].cost_usd == pytest.approx(0.355)
    line, message = produced.failed_lines[0]
    assert line.index == 1
    assert "0.43" in message and "no face" in message


def test_produce_stops_every_later_line_once_the_kling_pack_is_exhausted(tmp_path):
    from src.integrations.kling_video_client import KlingBalanceError

    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    video = _ReportingVideoClient(clip_bytes, costs=[None, None], errors=[KlingBalanceError("code 1102"), None])
    pipeline = DialoguePipeline(video, MockTTSClient(), str(tmp_path), image_client=MockReplicateImageClient())

    produced = pipeline.produce(_two_line_screenplay())

    assert len(video.calls) == 1  # the second line never reached Kling
    assert "Kling" in produced.failed_lines[1][1]


def _three_line_screenplay() -> DialogueScreenplay:
    return DialogueScreenplay(
        title="t", premise="p",
        characters=[Character(name="A", appearance="short dark hair")],
        lines=[DialogueLine(index=k, character="A", line=f"سطر {k}", action="a") for k in range(3)],
    )


def _scripted_reviewer(decisions: list[str]):
    calls = []

    def review(line, clip, attempt, can_redo):
        calls.append((line.index, attempt, can_redo, clip.cost_usd))
        return decisions[len(calls) - 1]

    return review, calls


def test_review_redo_pays_again_and_counts_the_rejected_clip(tmp_path):
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    video = _ReportingVideoClient(clip_bytes, costs=[0.4, 0.5, 0.6, 0.7], errors=[None] * 4)
    pipeline = DialoguePipeline(video, MockTTSClient(), str(tmp_path), image_client=MockReplicateImageClient())
    review, calls = _scripted_reviewer(["redo", "accept", "accept", "accept"])

    produced = pipeline.produce(_three_line_screenplay(), clip_reviewer=review)

    assert len(video.calls) == 4  # line 0 made twice
    assert calls[0][:3] == (0, 1, True) and calls[1][:3] == (0, 2, True)
    assert [c.cost_usd for c in produced.clips] == pytest.approx([0.5, 0.6, 0.7])
    assert produced.rejected_cost_usd == pytest.approx(0.4)
    portrait_cost = produced.portrait_cost_usd
    assert produced.total_cost_usd == pytest.approx(0.4 + 0.5 + 0.6 + 0.7 + portrait_cost)


def test_review_stop_spends_nothing_more(tmp_path):
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    video = _ReportingVideoClient(clip_bytes, costs=[0.4] * 3, errors=[None] * 3)
    pipeline = DialoguePipeline(video, MockTTSClient(), str(tmp_path), image_client=MockReplicateImageClient())
    review, _ = _scripted_reviewer(["stop"])

    produced = pipeline.produce(_three_line_screenplay(), clip_reviewer=review)

    assert len(video.calls) == 1
    assert len(produced.clips) == 1  # the clip shown before stopping is kept
    assert [l.index for l, _ in produced.failed_lines] == [1, 2]
    assert all("أوقفتَ" in m for _, m in produced.failed_lines)


def test_review_last_attempt_cannot_redo(tmp_path):
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    video = _ReportingVideoClient(clip_bytes, costs=[0.1] * 5, errors=[None] * 5)
    pipeline = DialoguePipeline(
        video, MockTTSClient(), str(tmp_path), image_client=MockReplicateImageClient(), max_review_attempts=2
    )
    # "redo" on the last attempt is not allowed and is treated as accept.
    review, calls = _scripted_reviewer(["redo", "redo", "accept", "accept"])

    produced = pipeline.produce(_three_line_screenplay(), clip_reviewer=review)

    assert calls[1][:3] == (0, 2, False)
    assert len(video.calls) == 4
    assert len(produced.clips) == 3


def test_review_failure_stops_production(tmp_path):
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)
    video = _ReportingVideoClient(clip_bytes, costs=[0.1] * 3, errors=[None] * 3)
    pipeline = DialoguePipeline(video, MockTTSClient(), str(tmp_path), image_client=MockReplicateImageClient())

    def broken(line, clip, attempt, can_redo):
        raise RuntimeError("telegram down")

    produced = pipeline.produce(_three_line_screenplay(), clip_reviewer=broken)

    assert len(video.calls) == 1
    assert "telegram down" in produced.failed_lines[0][1]
