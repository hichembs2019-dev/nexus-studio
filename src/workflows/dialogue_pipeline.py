"""Turns an approved DialogueScreenplay into real lip-synced clips and one
stitched episode, backed by FalVideoClient's real
fal-ai/wan/v2.2-14b/speech-to-video call - real characters speaking real
lines, the only pipeline this codebase has (see models.py's own
docstring: the old silent-T2V-plus-narration pipeline this one replaced
is deleted outright, not merely unused).

Clip length is NOT fixed - it is however long that line's own TTS audio
runs, because the model lip-syncs to the real audio track it is given.

Owner's instruction, 2026-10-19 ("اريدهم مثل المسلسلات التركية" - I want
them like Turkish TV dramas): DialogueLine.action only ever carried a
short, bare stage direction ("smiling, setting the table"), with nothing
pushing the actual visual generation toward any particular look.
CINEMATIC_STYLE_SUFFIX is appended to every real fal.ai prompt so every
clip reaches for the same warm, close, high-production-value television
drama look rather than a flat, undirected default.
"""
from __future__ import annotations

import base64
import mimetypes
import os
import subprocess

from src.integrations.fal_video_client import PRICE_PER_SECOND_USD, FalVideoClient
from src.integrations.image_client import PRICE_PER_MEGAPIXEL_USD, ImageClient
from src.integrations.tts_client import TTSClient
from src.models import Character, DialogueScreenplay, ProducedDialogueClip, ProducedDialogueEpisode

CINEMATIC_STYLE_SUFFIX = ", cinematic lighting, warm color grading, emotional close-up shot, high production value television drama"

# Owner's instruction, 2026-10-20, rejecting a fixed panel of six generic
# photos reused across every story forever ("لا تكون صور ثابتة الى الابد
# ... بل حسب القصة و السيناريو"): one real reference photo is generated
# per character, from that character's own `appearance` (inferred by the
# cast-writing LLM call from THIS premise, see dialogue_writer.py) -
# never two different stories' characters sharing the same face.
PORTRAIT_STYLE_SUFFIX = ", neutral friendly expression, looking directly at camera, soft studio lighting, plain neutral background, photorealistic, shot on a DSLR, sharp focus"

# Owner's instruction, 2026-10-20 ("نحذف الام و يبقى حوارها مكتوب على
# الشاشة... و كذلك نحذف الاب... لكي نستحقظ على المال" - remove the
# mother [as a speaking role], her dialogue stays written on screen;
# same for the father; to save money): a character.spoken=False line
# costs nothing real at all - no TTS, no fal.ai speech-to-video - and is
# rendered locally instead as on-screen text (Arabic, right-to-left,
# properly shaped - verified live with a real test render using this
# exact font before shipping) over a group photo of every captioned
# character, built once from their already-resolved real portraits.
#
# Real constraint checked, not assumed: ffmpeg's drawtext filter needs
# BOTH a font file with Arabic glyphs AND libfribidi (for correct
# right-to-left reordering) to render Arabic text correctly at all - the
# base python:3.12-slim image's `apt-get install ffmpeg` pulls in NO
# fonts whatsoever. fonts-noto-core (a real Debian/Ubuntu package,
# confirmed installable) ships NotoNaskhArabic-Regular.ttf, added to
# Dockerfile alongside this change - without it this path would either
# find no font at all or render Arabic as disconnected/wrong-order
# glyphs.
CAPTION_FONT_PATH = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"
CAPTION_CHARS_PER_SECOND = 15.0
CAPTION_MIN_SECONDS = 2.0


class DialoguePipelineError(Exception):
    pass


def _slug(text: str, default: str = "episode") -> str:
    # Real bug found by code review, 2026-10-21: this used to match only
    # `[a-z0-9]` - ASCII. Every real title and character name in this
    # entire project is Arabic, and Arabic letters never match that
    # class, so _slug("جسد بلا روح") and the equivalent inline regex for
    # a character's name both silently collapsed to the empty string and
    # fell back to the literal default ("episode"/"character") - THE
    # SAME fallback for every real screenplay and every real Arabic
    # character name, regardless of how different they actually are.
    # Reproduced directly: slug("جسد بلا روح ...") == "episode",
    # char_slug("فوزية") == char_slug("أحمد") == "character".
    #
    # The real consequence is severe, not cosmetic: _resolve_character_photo's
    # cache path is `{output_dir}/characters/{slug}_{char_slug}.jpg` and
    # starts with `if os.path.isfile(cache_path): return cache_path, 0.0, None`
    # - a cache HIT, not a failure, so nothing is reported. Once the
    # first real Arabic character photo is ever generated in a given
    # output_dir, EVERY later character - any name, any gender, any
    # story, forever - silently reuses that exact same file as a "cache
    # hit", with zero cost and zero error. This is the opposite of what
    # the owner has been demanding all session ("صور حسب القصة لا صور
    # ثابتة") - a real, story-specific photo became a single photo
    # shared by everyone, invisibly. The same collision also hit the
    # caption group image cache and every per-line output filename.
    #
    # str.isalnum() is Unicode-aware - True for Arabic letters, unlike
    # the ASCII-only regex class - so this now actually distinguishes
    # different Arabic titles and names instead of discarding them.
    out: list[str] = []
    prev_was_sep = True  # avoid a leading underscore
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
            prev_was_sep = False
        elif not prev_was_sep:
            out.append("_")
            prev_was_sep = True
    return "".join(out).strip("_") or default


def _data_uri(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime or 'application/octet-stream'};base64,{encoded}"


def _bytes_data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _ffprobe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr[-500:]}")
    return float(result.stdout.strip())


class DialoguePipeline:
    def __init__(
        self,
        fal_video: FalVideoClient,
        tts: TTSClient,
        output_dir: str,
        *,
        tts_lang: str = "ar-SY",
        resolution: str = "480p",
        max_cost_usd: float | None = None,
        image_client: ImageClient | None = None,
    ):
        self._fal_video = fal_video
        self._tts = tts
        self._output_dir = output_dir
        self._tts_lang = tts_lang
        self._resolution = resolution
        # Owner's standing instruction, 2026-10-21 ("احذف الستة صور و قلت
        # التوليد حسب القصة"): None here means production simply cannot
        # generate any real photo at all - every speaking character's
        # lines then fail and are reported (see produce() below), never
        # a generic/fixed placeholder face.
        self._image_client = image_client
        # Owner's instruction, 2026-10-20 ("قلت لك ضع له كود يشتغل و يسحب
        # المال الا عند صنع فيديو" - put code in that only draws money
        # when a video is actually made): a hard ceiling on total real
        # spend within one run, separate from the cost estimate already
        # shown before Approve - so a long or malformed screenplay can
        # never silently run the real bill past what was agreed to.
        self._max_cost_usd = max_cost_usd

    def produce(self, screenplay: DialogueScreenplay) -> ProducedDialogueEpisode:
        os.makedirs(self._output_dir, exist_ok=True)
        slug = _slug(screenplay.title)
        characters_by_name = {c.name: c for c in screenplay.characters}

        clips: list[ProducedDialogueClip] = []
        failed_lines: list[tuple] = []
        skipped_lines: list = []
        portrait_failures: list[tuple[str, str]] = []
        portrait_cost_usd = 0.0
        spent_so_far = 0.0

        # Pre-pass: one real, story-specific reference photo per speaking
        # character, generated once here and reused for every line that
        # character has below - never per line, and never regenerated on
        # a retry of this same screenplay (the cache path is keyed to
        # this episode's own slug + character name).
        #
        # Owner's standing instruction, 2026-10-21 ("احذف الستة صور و
        # قلت التوليد حسب القصة" - delete the six photos, I told you
        # generation must be per-story): resolved_photos now holds an
        # entry ONLY for a character whose real photo was actually
        # generated (or found already cached) - never a fallback. A
        # character missing from this dict has no usable photo at all,
        # and every spoken line of theirs below fails instead of ever
        # using a stand-in face.
        speaking_names = {l.character for l in screenplay.lines}
        resolved_photos: dict[str, str] = {}
        for character in screenplay.characters:
            if character.name not in speaking_names:
                continue
            if self._max_cost_usd is not None and spent_so_far >= self._max_cost_usd:
                portrait_failures.append(
                    (character.name, "تم الوصول للحد الأقصى للتكلفة قبل محاولة توليد صورة حقيقية لهذه الشخصية")
                )
                continue
            photo_path, cost, failure = self._resolve_character_photo(character, slug)
            spent_so_far += cost
            portrait_cost_usd += cost
            if photo_path is not None:
                resolved_photos[character.name] = photo_path
            if failure:
                portrait_failures.append((character.name, failure))

        # One shared group photo for every on-screen caption, built once
        # (free - just an ffmpeg hstack of portraits already resolved
        # above) from the real photos of whichever captioned characters
        # actually got one. A caption character with no resolved photo
        # is simply left out of this shared image (a pre-existing,
        # documented limitation - see README's "known limitation") -
        # caption lines cost $0 either way and are never gated on this.
        caption_character_names = sorted(
            {l.character for l in screenplay.lines if not l.spoken},
            key=lambda n: next(i for i, l in enumerate(screenplay.lines) if l.character == n),
        )
        resolved_caption_photos = [resolved_photos[n] for n in caption_character_names if n in resolved_photos]
        caption_group_image: str | None = None
        if resolved_caption_photos:
            caption_group_image = self._build_caption_group_image(resolved_caption_photos, slug)

        for line in screenplay.lines:
            character = characters_by_name.get(line.character)
            if character is None:
                raise DialoguePipelineError(f"Line {line.index} speaker {line.character!r} is not in the cast")

            if not line.spoken:
                # Real bug found by code review, 2026-10-21: the cost
                # ceiling check used to run BEFORE this branch, so a
                # caption line - genuinely $0, no TTS, no fal.ai call -
                # got silently skipped into skipped_lines whenever
                # earlier SPOKEN lines had already exhausted the real
                # ceiling, even though skipping it saves no real money
                # at all. The ceiling's own stated purpose ("a hard
                # ceiling on total REAL SPEND") never covered free
                # content - moved below so only spoken lines, which
                # actually cost something, are ever gated by it.
                if caption_group_image is None:
                    failed_lines.append(
                        (line, "لا توجد صورة حقيقية لأي من الشخصيات التي نصها مكتوب على الشاشة - لا توجد صورة احتياطية بأمر المالك")
                    )
                    continue
                try:
                    prefix = f"{slug}_line{line.index:03d}"
                    duration = max(CAPTION_MIN_SECONDS, len(line.line) / CAPTION_CHARS_PER_SECOND)
                    clip_path = os.path.join(self._output_dir, f"{prefix}.mp4")
                    self._render_caption_clip(caption_group_image, line.line, clip_path, duration)
                    clips.append(ProducedDialogueClip(line=line, local_path=clip_path, cost_usd=0.0))
                except Exception as exc:
                    failed_lines.append((line, str(exc)))
                continue

            if self._max_cost_usd is not None and spent_so_far >= self._max_cost_usd:
                # The ceiling was already reached by an earlier real,
                # already-paid line - this and every remaining SPOKEN
                # line is never attempted, so zero further real money is
                # spent. Caption lines above are never subject to this -
                # they cost nothing to begin with.
                skipped_lines.append(line)
                continue

            # Owner's standing instruction, 2026-10-21 ("احذف الستة صور و
            # قلت التوليد حسب القصة"): no fallback face exists any more -
            # a character whose real per-story photo could not be
            # generated (pre-pass above) simply has no usable photo, and
            # every spoken line of theirs fails here instead of ever
            # being produced with a stand-in face. Zero cost: this is
            # checked before TTS or any fal.ai call is made.
            photo_path = resolved_photos.get(character.name)
            if photo_path is None:
                reason = next((r for n, r in portrait_failures if n == character.name), "لم تُولَّد صورة حقيقية لهذه الشخصية")
                failed_lines.append(
                    (line, f"لا توجد صورة حقيقية لهذه الشخصية ({reason}) - لا توجد صورة احتياطية بأمر المالك")
                )
                continue

            # Real finding, 2026-10-19: a real 22-line run paid for and
            # successfully generated ~15 real clips (fal.ai's own
            # dashboard confirmed zero errors on its side), then one
            # line's real job ran past the client's own 600s wait - and
            # this used to let that exception propagate straight out of
            # produce(), discarding every clip already paid for and
            # already sitting finished on disk. Only the real,
            # money-spending network calls are caught here - a bad
            # screenplay (checks above) still stops everything, because
            # continuing would just repeat the same structural error.
            # Real bug found by code review, 2026-10-19: the single
            # except below used to let _format_failed_lines (conversation.py)
            # tell the owner every failed line's cost "was not counted" -
            # false whenever generate_speech_to_video() itself already
            # succeeded (and was billed) and only the LATER disk-write or
            # ffprobe step failed. billed tracks exactly that so the real
            # message can say which case actually happened.
            billed = False
            try:
                prefix = f"{slug}_line{line.index:03d}"
                audio_bytes = self._tts.synthesize(line.line, character.voice, lang=self._tts_lang)
                audio_path = os.path.join(self._output_dir, f"{prefix}.mp3")
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)

                image_data_uri = _data_uri(photo_path)
                audio_data_uri = _bytes_data_uri(audio_bytes, "audio/mpeg")

                clip_bytes = self._fal_video.generate_speech_to_video(
                    prompt=(line.action or f"{character.name} speaking") + CINEMATIC_STYLE_SUFFIX,
                    image_data_uri=image_data_uri,
                    audio_data_uri=audio_data_uri,
                    resolution=self._resolution,
                )
                billed = True
                clip_path = os.path.join(self._output_dir, f"{prefix}.mp4")
                with open(clip_path, "wb") as f:
                    f.write(clip_bytes)

                duration = _ffprobe_duration(clip_path)
                cost = duration * PRICE_PER_SECOND_USD[self._resolution]
                clips.append(ProducedDialogueClip(line=line, local_path=clip_path, cost_usd=cost))
                spent_so_far += cost
            except Exception as exc:
                message = str(exc)
                if billed:
                    message = f"[تم صرف مال حقيقي عند fal.ai قبل هذا الفشل - راجع لوحة fal.ai] {message}"
                failed_lines.append((line, message))
                continue

        # Real incident, 2026-10-20: a real run paid for and finished
        # 40.91s of real clips ($8.18, confirmed from fal.ai's own Usage
        # page), then this merge step raised - and because it sat outside
        # every try/except above, the exception escaped produce() entirely,
        # discarding every one of those real, already-paid, already-on-disk
        # clips. The owner was told "فشل التوليد الحقيقي ... لم يكتمل أي
        # مقطع" (nothing completed) - false, and worse than useless: it hid
        # the fact that real money had already bought something real that
        # was just never handed over. Only the stitch call itself needed
        # catching - a bad screenplay (checks at the top of this method)
        # still aborts everything, because continuing would just repeat
        # the same structural error on every line.
        stitched_path: str | None
        stitch_error: str | None = None
        try:
            stitched_path = self._stitch(clips, slug) if len(clips) > 1 else (clips[0].local_path if clips else None)
        except Exception as exc:
            stitched_path = None
            stitch_error = str(exc)
        return ProducedDialogueEpisode(
            screenplay=screenplay,
            clips=clips,
            stitched_path=stitched_path,
            failed_lines=failed_lines,
            stitch_error=stitch_error,
            skipped_lines=skipped_lines,
            portrait_cost_usd=portrait_cost_usd,
            portrait_failures=portrait_failures,
        )

    def _resolve_character_photo(self, character: Character, slug: str) -> tuple[str | None, float, str | None]:
        """Returns (path_or_None, real_cost_added_usd, failure_reason_or_None).
        Generates at most once ever per (this screenplay, this character) -
        a cached file on disk short-circuits every later call, including a
        retry of this same screenplay after an /edit.

        Owner's standing instruction, 2026-10-21 ("احذف الستة صور و قلت
        التوليد حسب القصة" - delete the six photos, I told you generation
        must be per-story): there is no fallback face any more. A real
        failure, a missing `appearance`, or no configured image client
        each return None - the caller (produce()) then fails that
        character's lines instead of ever using a stand-in photo."""
        char_slug = _slug(character.name, default="character")
        cache_dir = os.path.join(self._output_dir, "characters")
        cache_path = os.path.join(cache_dir, f"{slug}_{char_slug}.jpg")
        if os.path.isfile(cache_path):
            return cache_path, 0.0, None
        if self._image_client is None:
            return None, 0.0, "لا يوجد اتصال حقيقي بمولّد الصور (FAL_KEY غير مُهيَّأ)"
        if not character.appearance:
            return None, 0.0, "لا يوجد وصف لهذه الشخصية لتوليد صورة حقيقية منه"
        try:
            os.makedirs(cache_dir, exist_ok=True)
            prompt = f"photorealistic portrait photo of {character.appearance}" + PORTRAIT_STYLE_SUFFIX
            image_bytes = self._image_client.generate_image(prompt)
            with open(cache_path, "wb") as f:
                f.write(image_bytes)
            return cache_path, PRICE_PER_MEGAPIXEL_USD, None
        except Exception as exc:
            return None, 0.0, str(exc)

    def _build_caption_group_image(self, photo_paths: list[str], slug: str) -> str:
        """Built once per screenplay (not per caption line) - a real
        ffmpeg hstack of every captioned character's own real portrait,
        side by side. Pure local image work, zero real API cost."""
        out_path = os.path.join(self._output_dir, f"{slug}_caption_group.jpg")
        if os.path.isfile(out_path):
            return out_path
        if len(photo_paths) == 1:
            cmd = ["ffmpeg", "-y", "-i", photo_paths[0], out_path]
        else:
            inputs: list[str] = []
            for p in photo_paths:
                inputs += ["-i", p]
            cmd = [
                "ffmpeg", "-y", *inputs,
                "-filter_complex", f"hstack=inputs={len(photo_paths)}",
                out_path,
            ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg caption group image failed: {result.stderr[-2000:]}")
        return out_path

    def _render_caption_clip(self, group_image_path: str, text: str, out_path: str, duration: float) -> None:
        # Arabic drawtext needs an explicit font FILE (not a family name)
        # with real Arabic glyph coverage - verified live before shipping
        # this (see CAPTION_FONT_PATH's own comment). A single quote or
        # colon inside the line's own text would otherwise break out of
        # drawtext's text='...' argument - escaped the same way ffmpeg's
        # own docs show.
        escaped = text.replace("\\", "\\\\").replace("'", "’").replace(":", "\\:")
        drawtext = (
            f"drawtext=fontfile={CAPTION_FONT_PATH}:text='{escaped}':fontcolor=white:fontsize=36:"
            "box=1:boxcolor=black@0.6:boxborderw=20:x=(w-text_w)/2:y=h-th-60"
        )
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", group_image_path,
            "-vf", drawtext, "-t", f"{duration:.2f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg caption render failed: {result.stderr[-2000:]}")

    def _stitch(self, clips: list[ProducedDialogueClip], slug: str) -> str:
        list_path = os.path.join(self._output_dir, f"{slug}_dialogue_concat_list.txt")
        with open(list_path, "w") as f:
            f.write("\n".join(f"file '{os.path.abspath(c.local_path)}'" for c in clips))

        final_path = os.path.join(self._output_dir, f"{slug}_dialogue_stitched.mp4")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:v", "libx264", "-c:a", "aac", final_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg dialogue concat failed: {result.stderr[-2000:]}")
        return final_path
