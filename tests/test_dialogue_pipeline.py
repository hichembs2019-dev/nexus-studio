from __future__ import annotations

import os
import subprocess

import pytest

import base64

from src.integrations.fal_video_client import REAL_BILLED_PRICE_PER_SECOND_USD, MockFalVideoClient
from src.integrations.image_client import ImageGenerationError, MockImageClient, PRICE_PER_MEGAPIXEL_USD
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
    # portrait came back as - MockImageClient's own default scripted
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
# uses - a MockImageClient + a real `appearance` string - never a
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
    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    image_client = MockImageClient()
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
    fal_video = MockFalVideoClient()
    pipeline = DialoguePipeline(fal_video, MockTTSClient(), str(tmp_path))  # no image_client
    screenplay = DialogueScreenplay(
        title="Dinner", premise="p",
        characters=[Character(name="A", appearance="curly dark hair")],
        lines=[DialogueLine(index=0, character="A", line="one", action="a1")],
    )

    produced = pipeline.produce(screenplay)

    assert produced.portrait_cost_usd == 0.0
    assert produced.portrait_failures == [("A", "لا يوجد اتصال حقيقي بمولّد الصور (FAL_KEY غير مُهيَّأ)")]
    assert len(produced.clips) == 0
    assert len(fal_video.calls) == 0  # never even reached fal.ai - no photo to send it
    assert len(produced.failed_lines) == 1
    assert produced.total_cost_usd == 0.0


def test_produce_fails_a_characters_lines_when_it_has_no_appearance_description(tmp_path):
    image_client = MockImageClient()
    fal_video = MockFalVideoClient()
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
    assert len(fal_video.calls) == 0  # never reached fal.ai at all - no real photo existed to send
    assert len(produced.failed_lines) == 1


def test_produce_fails_the_characters_lines_when_real_portrait_generation_fails(tmp_path):
    class _FailingImageClient:
        def generate_image(self, prompt):
            raise ImageGenerationError("fal.ai image job failed: simulated real failure")

    fal_video = MockFalVideoClient()
    pipeline = DialoguePipeline(fal_video, MockTTSClient(), str(tmp_path), image_client=_FailingImageClient())
    screenplay = DialogueScreenplay(
        title="Dinner", premise="p",
        characters=[Character(name="A", appearance="curly dark hair")],
        lines=[DialogueLine(index=0, character="A", line="one", action="a1")],
    )

    produced = pipeline.produce(screenplay)

    assert len(produced.clips) == 0  # the line never ran - no fallback face, by the owner's order
    assert len(fal_video.calls) == 0  # fal.ai video was never called - no photo existed to send
    assert len(produced.failed_lines) == 1
    failed_line, message = produced.failed_lines[0]
    assert failed_line.index == 0
    assert "simulated real failure" in message
    assert produced.portrait_failures == [("A", "fal.ai image job failed: simulated real failure")]
    assert produced.portrait_cost_usd == 0.0
    assert produced.total_cost_usd == 0.0


def test_produce_single_line_calls_tts_then_fal_with_data_uris_and_saves_the_clip(tmp_path):
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient(scripted_bytes=b"\xff\xfbfake-mp3")
    image_client = MockImageClient(scripted_bytes=b"real-generated-portrait-bytes")
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
    # Real invoice evidence, 2026-10-21: fal.ai bills this model at the
    # 720p rate regardless of the "480p" actually requested - see
    # REAL_BILLED_PRICE_PER_SECOND_USD and docs/SOURCES.md.
    assert produced.clips[0].cost_usd == pytest.approx(1.0 * REAL_BILLED_PRICE_PER_SECOND_USD)


def test_produce_uses_the_real_price_per_second_explicitly_passed_in(tmp_path):
    # Real gap found by code review, 2026-10-21: DialoguePipeline used to
    # import and hard-code fal.ai's own REAL_BILLED_PRICE_PER_SECOND_USD
    # directly, so swapping in a different real provider (Replicate,
    # added the same night) would silently compute real cost at the
    # WRONG rate - exactly the "estimate doesn't match the real provider"
    # defect this project spent the whole evening fixing for fal.ai
    # itself. price_per_second_usd must now be honored explicitly,
    # distinct from both fal.ai's 0.10 and 0.20 constants so this can't
    # pass by coincidence.
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    pipeline = DialoguePipeline(
        fal_video, tts, str(tmp_path), image_client=MockImageClient(), price_per_second_usd=0.05
    )
    screenplay = DialogueScreenplay(
        title="T", premise="p",
        characters=[Character(name="A", appearance="a")],
        lines=[DialogueLine(index=0, character="A", line="one", action="a1")],
    )

    produced = pipeline.produce(screenplay)

    assert produced.clips[0].cost_usd == pytest.approx(1.0 * 0.05)


def test_produce_falls_back_to_a_default_prompt_when_action_is_empty(tmp_path):
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=MockImageClient())
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

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=MockImageClient())
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
    assert produced.total_cost_usd == pytest.approx(2 * 1.0 * REAL_BILLED_PRICE_PER_SECOND_USD + 2 * PRICE_PER_MEGAPIXEL_USD)


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

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=MockImageClient())
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
    assert produced.total_cost_usd == pytest.approx(2 * 1.0 * REAL_BILLED_PRICE_PER_SECOND_USD + 2 * PRICE_PER_MEGAPIXEL_USD)


def test_produce_stops_spending_once_the_hard_cost_ceiling_is_reached(tmp_path):
    # Owner's instruction, 2026-10-20 ("قلت لك ضع له كود يشتغل و يسحب
    # المال الا عند صنع فيديو" - put code in that only draws money when a
    # video is actually made): max_cost_usd is a hard ceiling on total
    # real spend within one run. A test that only checks the ceiling
    # exists as a number (never compared against anything) would pass
    # with the guard never actually stopping a real fal.ai call - this
    # asserts the THIRD line's real call never happens at all.
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    # Each clip costs 1.0s * REAL_BILLED_PRICE_PER_SECOND_USD (the real,
    # evidenced rate fal.ai actually bills - see docs/SOURCES.md) = $0.20.
    # The ceiling is only ever checked BEFORE a line is attempted (its
    # real cost isn't known until after fal.ai actually runs it), so a
    # ceiling below one line's cost is what proves the check fires at
    # all: the first line still goes through (nothing spent yet to
    # compare, beyond the one real portrait), every line after it must not.
    pipeline = DialoguePipeline(
        fal_video, tts, str(tmp_path), image_client=MockImageClient(), max_cost_usd=0.05
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
    assert produced.total_cost_usd == pytest.approx(REAL_BILLED_PRICE_PER_SECOND_USD + PRICE_PER_MEGAPIXEL_USD)


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
    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    pipeline = DialoguePipeline(
        fal_video, MockTTSClient(), str(tmp_path),
        image_client=MockImageClient(scripted_bytes=portrait_bytes), max_cost_usd=0.05,
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

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    image_client = MockImageClient(scripted_bytes=b"real-generated-portrait-bytes")
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=image_client)
    screenplay = DialogueScreenplay(
        title="Dinner",
        premise="p",
        characters=[Character(name="A", appearance="curly dark hair, slender build")],
        lines=[DialogueLine(index=0, character="A", line="one", action="a1")],
    )

    produced = pipeline.produce(screenplay)

    assert image_client.calls == ["photorealistic portrait photo of curly dark hair, slender build" + PORTRAIT_STYLE_SUFFIX]
    sent_image_b64 = fal_video.calls[0]["image_data_uri"].split(",", 1)[1]
    assert base64.b64decode(sent_image_b64) == b"real-generated-portrait-bytes"
    assert produced.portrait_cost_usd == pytest.approx(PRICE_PER_MEGAPIXEL_USD)
    assert produced.total_cost_usd == pytest.approx(REAL_BILLED_PRICE_PER_SECOND_USD + PRICE_PER_MEGAPIXEL_USD)


def test_produce_reuses_the_cached_portrait_without_regenerating_on_a_second_call(tmp_path):
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    image_client = MockImageClient()
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


def test_produce_portrait_generation_respects_the_hard_cost_ceiling(tmp_path):
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    image_client = MockImageClient()
    # The ceiling allows the FIRST portrait through (nothing spent yet to
    # compare against) but must block the second real portrait call -
    # same "checked before, not after" principle as the per-line ceiling.
    pipeline = DialoguePipeline(
        fal_video, MockTTSClient(), str(tmp_path),
        image_client=image_client, max_cost_usd=PRICE_PER_MEGAPIXEL_USD / 2,
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

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=MockImageClient(scripted_bytes=portrait_bytes))
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
    assert produced.total_cost_usd == pytest.approx(REAL_BILLED_PRICE_PER_SECOND_USD + 2 * PRICE_PER_MEGAPIXEL_USD)


@pytest.mark.skipif(not _ARABIC_FONT_AVAILABLE, reason="fonts-noto-core not installed on this machine - Dockerfile installs it for the real deployment")
def test_produce_captions_a_line_even_when_its_own_character_has_no_photo(tmp_path):
    # Documented limitation (README): the caption group photo is built
    # from whichever captioned characters' photos actually resolved - a
    # caption LINE still costs $0 and still renders even if its own
    # character's portrait failed, as long as at least one other
    # captioned character's photo is available to show.
    portrait_bytes = _real_jpeg_bytes(str(tmp_path / "portrait.jpg"))
    fal_video = MockFalVideoClient()
    pipeline = DialoguePipeline(fal_video, MockTTSClient(), str(tmp_path), image_client=MockImageClient(scripted_bytes=portrait_bytes))
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
    fal_video = MockFalVideoClient()
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
        MockFalVideoClient(), MockTTSClient(), str(tmp_path),
        image_client=MockImageClient(scripted_bytes=portrait_bytes),
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
        MockFalVideoClient(), MockTTSClient(), str(tmp_path), image_client=MockImageClient(scripted_bytes=portrait_a)
    )
    screenplay_a = DialogueScreenplay(
        title="Same Title", premise="p",
        characters=[Character(name="A", appearance="a")],
        lines=[DialogueLine(index=0, character="A", line="نص أ", action="", spoken=False)],
    )
    pipeline_a.produce(screenplay_a)

    portrait_b = _real_jpeg_bytes(str(tmp_path / "b.jpg"))
    pipeline_b = DialoguePipeline(
        MockFalVideoClient(), MockTTSClient(), str(tmp_path), image_client=MockImageClient(scripted_bytes=portrait_b)
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
    pipeline = DialoguePipeline(MockFalVideoClient(), MockTTSClient(), str(tmp_path))
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

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes, fail_at_calls=frozenset({1}))
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=MockImageClient())
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
    assert produced.total_cost_usd == pytest.approx(2 * 1.0 * REAL_BILLED_PRICE_PER_SECOND_USD + PRICE_PER_MEGAPIXEL_USD)


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

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=MockImageClient())
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

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient(scripted_bytes=mp3_bytes)
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), image_client=MockImageClient())
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
