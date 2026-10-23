"""Turns an approved DialogueScreenplay into real lip-synced clips and one
stitched episode, backed by ReplicateVideoClient's real
wan-video/wan-2.2-s2v speech-to-video call - real characters speaking real
lines, the only pipeline this codebase has (see models.py's own
docstring: the old silent-T2V-plus-narration pipeline this one replaced
is deleted outright, not merely unused).

Owner's standing instruction, 2026-10-21 ("لا تضع السراق ... احذفهم من
الكود" - don't put the thieves [fal.ai] in, remove them from the code,
repeated twice more with "خذ وقتك ... هذا ما طلبته منك بالظبط"): fal.ai's
own client (`fal_video_client.py`, `image_client.py`) is deleted outright
from this repository, after its own real billing could not be trusted
(see docs/SOURCES.md, the 720p/480p finding). Replicate is the only real
video/image provider this pipeline can be wired to.

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

from src.integrations.multitalk_video_client import MultiTalkVideoClient
from src.integrations.replicate_image_client import PRICE_PER_IMAGE_USD_UNVERIFIED, ReplicateImageClient
from src.integrations.replicate_video_client import PROVISIONAL_PRICE_PER_SECOND_USD, ReplicateVideoClient
from src.integrations.tts_client import TTSClient
from src.models import Character, DialogueLine, DialogueScreenplay, ProducedDialogueClip, ProducedDialogueEpisode

CINEMATIC_STYLE_SUFFIX = ", cinematic lighting, warm color grading, emotional close-up shot, high production value television drama"

# Owner's instruction, 2026-10-20, rejecting a fixed panel of six generic
# photos reused across every story forever ("لا تكون صور ثابتة الى الابد
# ... بل حسب القصة و السيناريو"): one real reference photo is generated
# per character, from that character's own `appearance` (inferred by the
# cast-writing LLM call from THIS premise, see dialogue_writer.py) -
# never two different stories' characters sharing the same face.
#
# Real finding, 2026-10-22: the owner compared a real produced clip
# against the first real video this project ever made (two people
# together in a real kitchen, a bowl of rice on the table between them)
# and named the real difference precisely: "الفرق بينهم في المشهد" (the
# difference between them is in the SCENE). It was never about how the
# lines were distributed (already fixed, see _MAX_SINGLE_CHARACTER_LINE_SHARE
# in dialogue_writer.py) - every reference photo was generated as an
# isolated studio headshot against a flat, empty background, by this
# constant's own old wording ("soft studio lighting, plain neutral
# background"). No matter how real the lip-sync or how balanced the
# dialogue, a floating head in front of nothing can never read as a real
# film scene. Rewritten to place the character inside a real, lived-in
# location consistent with the story - still one still photo, still no
# second person in the same frame (that would need a genuinely different,
# unproven model - see docs/SOURCES.md), but a real environment instead
# of an empty backdrop, which is the one concrete, low-risk improvement
# available without a new model or new real-money risk.
PORTRAIT_STYLE_SUFFIX = ", natural expression, looking directly at camera, photorealistic, shot on a DSLR, sharp focus, cinematic lighting - set within a real, lived-in location that fits the story (a home, a workplace, outdoors, wherever the story actually places them), never a plain studio backdrop or neutral background"

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

# Real bug found 2026-10-22 (see Character.gender's own comment,
# models.py): the portrait prompt used to carry no gender signal at all,
# only `appearance`'s free-text sentence - a real produced episode gave
# a character written and spoken-of as "الشاب" (the young man) a
# woman's face. The cast-writing LLM already infers "male"/"female" for
# every character; this just stops that real inference from being
# thrown away before it reaches the image prompt.
_GENDER_NOUN = {"male": "a man", "female": "a woman"}


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


def _ffprobe_dimensions(path: str) -> tuple[int, int]:
    """Real width/height of an already-generated image, measured - never
    assumed. Needed to build a real two-speaker reference's per-speaker
    masks at the exact real pixel size MultiTalk's own `ref_target_masks`
    input requires (see multitalk_prompt_builder.py's own docstring for
    why that input is required at all, not guessed from a fixed constant
    that could silently drift from whatever size Replicate's portrait
    model actually returns."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or "x" not in result.stdout:
        raise RuntimeError(f"ffprobe dimensions failed on {path}: {result.stderr[-500:]}")
    width_str, height_str = result.stdout.strip().split("x")
    return int(width_str), int(height_str)


class DialoguePipeline:
    def __init__(
        self,
        fal_video: ReplicateVideoClient,
        tts: TTSClient,
        output_dir: str,
        *,
        tts_lang: str = "ar-SY",
        resolution: str = "480p",
        max_cost_usd: float | None = None,
        image_client: ReplicateImageClient | None = None,
        # Real wiring, 2026-10-23, once MultiTalk's two real crashes were
        # fixed and a real two-speaker video was proven end to end (see
        # docs/SOURCES.md, "MultiTalk two-speaker generation proven real,
        # end to end"): None (the default) keeps every existing behavior
        # completely unchanged - every spoken line still produces its own
        # isolated single-speaker clip, exactly as before. Only when this
        # is actually configured (src/app.py's own explicit gate) does
        # produce() look for real consecutive two-character exchanges to
        # route through one real shared-scene MultiTalk clip instead -
        # see this class's own produce() docstring for exactly which
        # exchanges qualify and why.
        multitalk_client: MultiTalkVideoClient | None = None,
        # Real finding, 2026-10-21 ("و لماذا تتسرب الأسعار ... ألم تقل لك
        # من قبل؟"): the real rate for whichever client is actually wired
        # in must be passed explicitly by the caller (src/app.py) - never
        # hard-coded here, which is exactly the "wrong rate used without
        # anyone noticing" defect this project spent the whole evening
        # fixing for fal.ai's own rate before fal.ai was deleted outright.
        price_per_second_usd: float = PROVISIONAL_PRICE_PER_SECOND_USD,
        # Owner's instruction, 2026-10-21 ("قم بالتنظيف ... كثرة الكودات
        # اصبحت تتداخل في بعضها" - clean up, too much code has started
        # to overlap): same reasoning as price_per_second_usd above, for
        # the portrait cost component.
        price_per_image_usd: float = PRICE_PER_IMAGE_USD_UNVERIFIED,
    ):
        self._fal_video = fal_video
        self._tts = tts
        self._output_dir = output_dir
        self._tts_lang = tts_lang
        self._resolution = resolution
        self._price_per_second_usd = price_per_second_usd
        self._price_per_image_usd = price_per_image_usd
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
        self._multitalk_client = multitalk_client

    def save_character_photo(self, title: str, character_name: str, image_bytes: bytes) -> str:
        """Owner's instruction, 2026-10-21 ("الصور ضعها انت الان و هكذا
        لا ندفع شيء" - you put the photos in yourself, so we don't pay
        anything): writes image_bytes directly to the EXACT cache path
        _resolve_character_photo() checks first
        (`{output_dir}/characters/{slug(title)}_{slug(character_name)}.jpg`) -
        the same real cache-hit path a real `produce()` run already
        trusts unconditionally. A photo saved this way is picked up by
        the next real production of this screenplay exactly like a
        previously-generated one would be: zero cost, zero real
        generation call, because the pipeline never sees a reason to
        make one. Overwrites any existing cached photo at that path
        without warning - the caller (runner.py's /photo handler) is the
        one place that decides this is wanted."""
        slug = _slug(title)
        char_slug = _slug(character_name, default="character")
        cache_dir = os.path.join(self._output_dir, "characters")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{slug}_{char_slug}.jpg")
        with open(cache_path, "wb") as f:
            f.write(image_bytes)
        return cache_path

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
            photo_path, cost, failure = self._resolve_character_photo(character, slug, screenplay.premise)
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
        included_caption_names = [n for n in caption_character_names if n in resolved_photos]
        resolved_caption_photos = [resolved_photos[n] for n in included_caption_names]
        caption_group_image: str | None = None
        if resolved_caption_photos:
            caption_group_image = self._build_caption_group_image(resolved_caption_photos, slug, included_caption_names)

        # Real design point, 2026-10-23: scenes are not produced as one
        # real unit, even when a real MultiTalk client is configured.
        # MultiTalk's own "add" multi_audio_type - proven live via the
        # real output waveform's own length (`final waveform shape ...
        # expected sum of raw`, see docs/SOURCES.md) - is literal
        # concatenation: speaker 1's WHOLE track, then speaker 2's WHOLE
        # track, back to back. A real scene with more than one real
        # back-and-forth exchange (فوزية / جويدة / فوزية / ...) fed to ONE
        # MultiTalk call this way would play every one of فوزية's lines
        # first, then every one of جويدة's - scrambling the real written
        # order, not representing it. So pairing is scoped to exactly
        # TWO CONSECUTIVE real lines, same real scene, two DIFFERENT real
        # speakers - the one shape "add" mode actually reproduces
        # correctly (line A's real audio, then line B's real audio, in
        # the real order they were written) - never a whole multi-turn
        # scene in one call.
        i = 0
        n = len(screenplay.lines)
        while i < n:
            line = screenplay.lines[i]
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
                    i += 1
                    continue
                try:
                    prefix = f"{slug}_line{line.index:03d}"
                    duration = max(CAPTION_MIN_SECONDS, len(line.line) / CAPTION_CHARS_PER_SECOND)
                    clip_path = os.path.join(self._output_dir, f"{prefix}.mp4")
                    self._render_caption_clip(caption_group_image, line.line, clip_path, duration)
                    clips.append(ProducedDialogueClip(line=line, local_path=clip_path, cost_usd=0.0))
                except Exception as exc:
                    failed_lines.append((line, str(exc)))
                i += 1
                continue

            next_line = screenplay.lines[i + 1] if i + 1 < n else None
            pairs_with_next = (
                self._multitalk_client is not None
                and next_line is not None
                and next_line.spoken
                and next_line.scene == line.scene
                and next_line.character != line.character
            )
            next_character = characters_by_name.get(next_line.character) if pairs_with_next else None
            if pairs_with_next and next_character is None:
                raise DialoguePipelineError(f"Line {next_line.index} speaker {next_line.character!r} is not in the cast")
            photo_path = resolved_photos.get(character.name)
            next_photo_path = resolved_photos.get(next_character.name) if pairs_with_next else None
            pairs_with_next = pairs_with_next and photo_path is not None and next_photo_path is not None

            if pairs_with_next:
                if self._max_cost_usd is not None and spent_so_far >= self._max_cost_usd:
                    skipped_lines.append(line)
                    skipped_lines.append(next_line)
                    i += 2
                    continue
                billed = False
                try:
                    audio_a = self._tts.synthesize(line.line, character.voice, lang=self._tts_lang)
                    audio_b = self._tts.synthesize(next_line.line, next_character.voice, lang=self._tts_lang)
                    ref_path, mask_a_path, mask_b_path = self._build_two_speaker_reference(
                        photo_path, next_photo_path, slug,
                        _slug(character.name, default="character"), _slug(next_character.name, default="character"),
                    )
                    with open(ref_path, "rb") as f:
                        ref_bytes = f.read()
                    with open(mask_a_path, "rb") as f:
                        mask_a_bytes = f.read()
                    with open(mask_b_path, "rb") as f:
                        mask_b_bytes = f.read()

                    prompt = (
                        f"{character.name} and {next_character.name} talking to each other in a real scene"
                    ) + CINEMATIC_STYLE_SUFFIX
                    clip_bytes = self._multitalk_client.generate_two_speaker_video(
                        prompt=prompt,
                        reference_image_bytes=ref_bytes, reference_image_mime="image/jpeg",
                        speaker_1_audio_bytes=audio_a, speaker_1_audio_mime="audio/mpeg", speaker_1_mask_bytes=mask_a_bytes,
                        speaker_2_audio_bytes=audio_b, speaker_2_audio_mime="audio/mpeg", speaker_2_mask_bytes=mask_b_bytes,
                    )
                    billed = True
                    prefix = f"{slug}_line{line.index:03d}_{next_line.index:03d}_multitalk"
                    clip_path = os.path.join(self._output_dir, f"{prefix}.mp4")
                    with open(clip_path, "wb") as f:
                        f.write(clip_bytes)
                    try:
                        duration = _ffprobe_duration(clip_path)
                        cost = duration * self._price_per_second_usd
                        clips.append(ProducedDialogueClip(
                            line=line, local_path=clip_path, cost_usd=cost, covered_lines=[line, next_line]
                        ))
                        spent_so_far += cost
                    except Exception as probe_exc:
                        clips.append(ProducedDialogueClip(
                            line=line, local_path=clip_path, cost_usd=0.0, covered_lines=[line, next_line]
                        ))
                        failed_lines.append((
                            line,
                            f"[تم صرف مال حقيقي وحُفظ الفيديو الحقيقي فعليًا - مرفق أدناه - لكن تعذّر قياس مدته "
                            f"الفعلية، فلم تُحتسب تكلفته في المجموع] {probe_exc}",
                        ))
                except Exception as exc:
                    message = str(exc)
                    if billed:
                        message = f"[تم صرف مال حقيقي عند الإنتاج الذاتي قبل هذا الفشل] {message}"
                    failed_lines.append((line, message))
                    failed_lines.append((next_line, message))
                i += 2
                continue

            if self._max_cost_usd is not None and spent_so_far >= self._max_cost_usd:
                # The ceiling was already reached by an earlier real,
                # already-paid line - this and every remaining SPOKEN
                # line is never attempted, so zero further real money is
                # spent. Caption lines above are never subject to this -
                # they cost nothing to begin with.
                skipped_lines.append(line)
                i += 1
                continue

            # Owner's standing instruction, 2026-10-21 ("احذف الستة صور و
            # قلت التوليد حسب القصة"): no fallback face exists any more -
            # a character whose real per-story photo could not be
            # generated (pre-pass above) simply has no usable photo, and
            # every spoken line of theirs fails here instead of ever
            # being produced with a stand-in face. Zero cost: this is
            # checked before TTS or any fal.ai call is made.
            if photo_path is None:
                reason = next((r for n, r in portrait_failures if n == character.name), "لم تُولَّد صورة حقيقية لهذه الشخصية")
                failed_lines.append(
                    (line, f"لا توجد صورة حقيقية لهذه الشخصية ({reason}) - لا توجد صورة احتياطية بأمر المالك")
                )
                i += 1
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

                # Owner's standing instruction, 2026-10-21 ("لا يجب ان
                # يفشل شيء لاني ادفع مالا عند الفشل" - nothing may fail,
                # because I pay money when it fails): by this point the
                # real video was already generated and billed by fal.ai,
                # and its real bytes are genuinely sitting on disk at
                # clip_path - only measuring its duration (to compute its
                # real cost) is left. If THAT alone fails (a corrupt
                # response, a transient ffprobe glitch), the old code
                # discarded this real, paid, already-downloaded file
                # completely - the owner would have paid fal.ai for a
                # video that exists on the server and never reaches them,
                # not even as a path to retrieve manually. The real audio
                # file this video is lip-synced to was written successfully
                # moments earlier and is lip-synced 1:1 to the video, so
                # probing IT is a real, measured (not guessed) stand-in
                # for the video's own duration when the video probe
                # itself fails.
                try:
                    duration = _ffprobe_duration(clip_path)
                    # Real cost tracking uses self._price_per_second_usd -
                    # the real rate for whichever provider is ACTUALLY
                    # wired into self._fal_video, passed explicitly by the
                    # caller (src/app.py). See this class's own __init__
                    # docstring comment for why this is never a hard-coded
                    # import any more.
                    cost = duration * self._price_per_second_usd
                    clips.append(ProducedDialogueClip(line=line, local_path=clip_path, cost_usd=cost))
                    spent_so_far += cost
                except Exception as probe_exc:
                    try:
                        duration = _ffprobe_duration(audio_path)
                        cost = duration * self._price_per_second_usd
                        clips.append(ProducedDialogueClip(line=line, local_path=clip_path, cost_usd=cost))
                        spent_so_far += cost
                    except Exception:
                        # Could not measure either real file - the real
                        # paid video is still delivered rather than
                        # discarded (the whole point of this fix), but its
                        # exact real cost genuinely cannot be measured, so
                        # it is reported as unmeasured (not guessed with an
                        # invented number) and counted as $0 in the total -
                        # a safety-margin understatement, never an
                        # overstatement, with the real reason surfaced so
                        # the owner can check fal.ai's own dashboard.
                        clips.append(ProducedDialogueClip(line=line, local_path=clip_path, cost_usd=0.0))
                        failed_lines.append(
                            (line, f"[تم صرف مال حقيقي وحُفظ الفيديو الحقيقي فعليًا - مرفق أدناه - لكن تعذّر قياس مدته الفعلية، فلم تُحتسب تكلفته في المجموع؛ راجع لوحة fal.ai للتكلفة الدقيقة] {probe_exc}")
                        )
            except Exception as exc:
                message = str(exc)
                if billed:
                    message = f"[تم صرف مال حقيقي عند fal.ai قبل هذا الفشل - راجع لوحة fal.ai] {message}"
                failed_lines.append((line, message))
                i += 1
                continue
            i += 1

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

    def _resolve_character_photo(
        self, character: Character, slug: str, premise: str
    ) -> tuple[str | None, float, str | None]:
        """Returns (path_or_None, real_cost_added_usd, failure_reason_or_None).
        Generates at most once ever per (this screenplay, this character) -
        a cached file on disk short-circuits every later call, including a
        retry of this same screenplay after an /edit.

        Owner's standing instruction, 2026-10-21 ("احذف الستة صور و قلت
        التوليد حسب القصة" - delete the six photos, I told you generation
        must be per-story): there is no fallback face any more. A real
        failure, a missing `appearance`, or no configured image client
        each return None - the caller (produce()) then fails that
        character's lines instead of ever using a stand-in photo.

        `premise` (the real screenplay's own story text) is passed in so
        the real image model has something to place the character INSIDE
        - see PORTRAIT_STYLE_SUFFIX's own 2026-10-22 comment for why a
        studio headshot with nothing behind it can never read as a real
        film scene, however balanced the dialogue is."""
        char_slug = _slug(character.name, default="character")
        cache_dir = os.path.join(self._output_dir, "characters")
        cache_path = os.path.join(cache_dir, f"{slug}_{char_slug}.jpg")
        if os.path.isfile(cache_path):
            return cache_path, 0.0, None
        if self._image_client is None:
            return None, 0.0, "لا يوجد اتصال حقيقي بمولّد الصور (REPLICATE_API_TOKEN غير مُهيَّأ)"
        if not character.appearance:
            return None, 0.0, "لا يوجد وصف لهذه الشخصية لتوليد صورة حقيقية منه"
        try:
            os.makedirs(cache_dir, exist_ok=True)
            # Capped, not the full premise verbatim: this only needs to
            # tell the image model WHERE the character real is (a
            # kitchen, a hospital room, a street), not retell the whole
            # story - and a very long /full premise would otherwise bloat
            # every single real image prompt for no real benefit.
            scene_context = premise[:300]
            gender_noun = _GENDER_NOUN.get(character.gender, "a person")
            prompt = (
                f"photorealistic portrait photo of {gender_noun}, {character.appearance}, in a scene from this "
                f"real story: {scene_context}"
            ) + PORTRAIT_STYLE_SUFFIX
            image_bytes = self._image_client.generate_image(prompt)
            with open(cache_path, "wb") as f:
                f.write(image_bytes)
            return cache_path, self._price_per_image_usd, None
        except Exception as exc:
            return None, 0.0, str(exc)

    def _build_caption_group_image(self, photo_paths: list[str], slug: str, character_names: list[str]) -> str:
        """Built once per screenplay (not per caption line) - a real
        ffmpeg hstack of every captioned character's own real portrait,
        side by side. Pure local image work, zero real API cost.

        Real bug found by code review, 2026-10-21: the cache path used to
        be keyed only by `slug` (the screenplay's own title), with no
        trace of WHICH characters were actually in the group photo. A
        retry of the same screenplay (same title, same slug - the exact,
        already-documented, deliberate reason character photos ARE cached
        by slug) after an /edit or /caption that changed which characters
        are captioned would silently reuse the OLD group image - the
        wrong characters shown in every caption frame, with no error at
        all. The cache key now includes every included character's own
        name, the same `_slug()` the per-character photo cache already
        uses, so a genuinely different caption roster can never collide
        with a stale file."""
        composition_key = "_".join(_slug(n, default="c") for n in character_names)
        out_path = os.path.join(self._output_dir, f"{slug}_caption_group_{composition_key}.jpg")
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

    def _build_two_speaker_reference(
        self, photo_path_1: str, photo_path_2: str, slug: str, char_slug_1: str, char_slug_2: str
    ) -> tuple[str, str, str]:
        """Returns (composite_reference_image_path, speaker_1_mask_path,
        speaker_2_mask_path) for a real two-speaker MultiTalk call -
        built once per (this screenplay, these two characters, in this
        order) and cached, the same caching posture every other real
        asset in this file already uses.

        Real, deliberate design choice, 2026-10-23: rather than a face-
        detection dependency this project does not have (and the real
        risk of a wrong detection silently mis-assigning which speaker's
        audio drives which face), each character's ALREADY-GENERATED,
        already-cached real portrait (see `_resolve_character_photo`) is
        placed at a fixed, KNOWN position - left half for the first
        speaker, right half for the second, the exact `hstack` technique
        `_build_caption_group_image` already uses - so the real mask for
        each speaker is simple, exact arithmetic on the composite's own
        real measured dimensions, never a guess and never a detection
        that could be wrong. Honest real limitation, not hidden: this is
        two separate real solo photos placed side by side, not one
        photo of both characters together in a single real shot - a
        visibly different, less cinematic composite than the hand-picked
        real movie frame the first live MultiTalk proof used. Upgrading
        this to one real composed shot (a single image-generation call
        describing both characters together) is a tracked follow-up, not
        bundled into this first real wiring."""
        out_dir = os.path.join(self._output_dir, "multitalk_refs")
        os.makedirs(out_dir, exist_ok=True)
        composite_path = os.path.join(out_dir, f"{slug}_{char_slug_1}_{char_slug_2}.jpg")
        mask_1_path = os.path.join(out_dir, f"{slug}_{char_slug_1}_{char_slug_2}_mask1.png")
        mask_2_path = os.path.join(out_dir, f"{slug}_{char_slug_1}_{char_slug_2}_mask2.png")

        if not os.path.isfile(composite_path):
            cmd = [
                "ffmpeg", "-y", "-i", photo_path_1, "-i", photo_path_2,
                "-filter_complex", "hstack=inputs=2", composite_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg two-speaker reference composite failed: {result.stderr[-2000:]}")

        if not os.path.isfile(mask_1_path) or not os.path.isfile(mask_2_path):
            width, height = _ffprobe_dimensions(composite_path)
            tile_width = width // 2
            for mask_path, x_start, x_width in (
                (mask_1_path, 0, tile_width),
                (mask_2_path, tile_width, width - tile_width),
            ):
                cmd = [
                    "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}",
                    "-vf", f"drawbox=x={x_start}:y=0:w={x_width}:h={height}:color=white:t=fill",
                    "-frames:v", "1", mask_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg two-speaker mask build failed: {result.stderr[-2000:]}")

        return composite_path, mask_1_path, mask_2_path

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
