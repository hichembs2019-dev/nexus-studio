"""Dialogue-native episode shapes - real characters speaking real lines.

Owner's standing instruction, 2026-10-21 ("لماذا تصر ان يوجد الراوي في
الكودات ... احذف الراوي للابد الا بطلب مني" - why do you insist the
narrator exists in the code, delete it permanently unless I ask for it
myself): this module used to also define the OLD silent-T2V-plus-
narration pipeline's shapes (Screenplay/ScreenplayScene, Scene, Episode,
ProducedClip, ProducedEpisode, and the CLIP_*/GPU_HOURLY_COST_USD
constants they used) - the real "راوي" (narrator) mechanism: one prose
screenplay, scenes without speakers, a narration track read over silent
video. The live bot stopped reaching any of that code on 2026-10-19 (see
docs/SOURCES.md), but the classes themselves - and the WriterAgent/
DirectorAgent/EpisodePipeline/ComfyUIClient/RemoteExecClient code built
around them - kept existing, unused, in this repository until this
commit. Deleted outright, not merely made unreachable, per that
instruction: the narrator no longer exists anywhere in this codebase,
only `/new` and `/full`'s real dialogue path (DialogueWriterAgent,
DialoguePipeline) below.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Character:
    """A speaking role. Wan 2.2's speech-to-video call (see
    fal_video_client.py) takes one reference image per clip, and reusing
    the same real image for every line this character speaks is what
    keeps their face the same person scene to scene - DialoguePipeline
    generates that real image itself, from `appearance` below, and
    refuses to produce a line for a character it could not generate one
    for (see DialoguePipeline's own docstring).

    Owner's standing instruction, 2026-10-21 ("احذف الستة صور و قلت
    التوليد حسب القصة" - delete the six photos, I told you generation
    must be per-story): there used to also be a `reference_image_path`
    field here, always pre-filled at screenplay-write time with one of
    six generic placeholder illustrations (`character_photos.py`'s
    `resolve_placeholder_photo`) and reused as DialoguePipeline's silent
    fallback whenever real per-story generation failed or was
    unavailable. Deleted outright, field and all: there is no longer any
    fixed/generic photo anywhere in this codebase, for any reason - a
    character either gets a real photo generated from its own
    `appearance` during production, or that character's lines fail and
    are reported, never a stand-in face.
    """

    name: str
    # Owner's instruction, 2026-10-21: switched back to Tunisian - see
    # character_photos.py's own docstring for the full real history
    # (Tunisian -> Syrian on 2026-10-19, reversed back on 2026-10-21
    # after a real produced episode's audio proved the Syrian dialect
    # genuinely unclear to the owner, contradicting the earlier reasoning).
    voice: str = "ar-TN-HediNeural"
    # Owner's instruction, 2026-10-20, rejecting a fixed panel of six
    # generic photos reused across every story forever ("لا تكون صور
    # ثابتة الى الابد ... بل حسب القصة و السيناريو" - not fixed forever,
    # but according to the story and screenplay): a short visual
    # description (hair, build, distinguishing features, age) inferred
    # from THIS premise by the cast-writing LLM call - the ONLY source
    # DialoguePipeline ever generates a character's real photo from.
    appearance: str = ""


@dataclass
class DialogueLine:
    """One beat of a DialogueScreenplay: a single character speaking a
    single line, the unit DialoguePipeline turns into one real
    speech-to-video clip (audio = this line's own TTS, image = the
    speaking character's reference photo). `action` is the visual-only
    stage direction for that moment (expression, gesture, what else is in
    frame) - never dialogue text itself, since the S2V model lip-syncs the
    audio and would just as happily render the action text as more speech
    if it leaked in here."""

    index: int
    character: str
    line: str
    action: str = ""
    # Owner's instruction, 2026-10-20 ("نحذف الام و يبقى حوارها... مكتوب
    # على الشاشة" - remove the mother [as a speaking role], her dialogue
    # stays as on-screen text): a real, concrete way to cut cost on a line
    # that doesn't need to be heard - spoken=False skips TTS and the real
    # fal.ai speech-to-video call entirely (zero real cost) and renders
    # the line as an on-screen caption instead (see
    # dialogue_pipeline.py's _render_caption_clip). Defaults to True so
    # every existing screenplay/line keeps producing a real spoken clip
    # exactly as before.
    spoken: bool = True


@dataclass
class DialogueScreenplay:
    """The dialogue-native counterpart to Screenplay (models.py) - real
    characters speaking real lines, the actual ask ("مسلسلات وأفلام ... و
    نشرها", series/films with real characters, not a narrated
    documentary). The owner reads/edits `lines` directly, the same human
    gate Screenplay already has, before DialoguePipeline spends anything."""

    title: str
    premise: str
    characters: list[Character] = field(default_factory=list)
    lines: list[DialogueLine] = field(default_factory=list)


@dataclass
class ProducedDialogueClip:
    line: DialogueLine
    local_path: str
    cost_usd: float


@dataclass
class ProducedDialogueEpisode:
    screenplay: DialogueScreenplay
    clips: list[ProducedDialogueClip]
    stitched_path: str | None
    # Real finding, 2026-10-19: a real 22-line run paid for and
    # successfully generated ~15 real clips (fal.ai's own dashboard:
    # "Errors in the last 7 days: 0"), then one line's job ran past the
    # client's own 600s wait - and DialoguePipeline.produce() used to
    # raise and discard the whole episode right there, throwing away
    # every clip already paid for. Now it keeps going: each (line, real
    # error message) that failed lands here instead of aborting
    # everything, so the owner gets back whatever real money actually
    # bought, not nothing.
    failed_lines: list[tuple[DialogueLine, str]] = field(default_factory=list)
    # Real incident, 2026-10-20: a real run billed $8.18 (40.91s of real
    # completed fal.ai output, confirmed from fal.ai's own Usage page) for
    # clips that genuinely finished - then the final ffmpeg merge step
    # failed, and that exception used to escape DialoguePipeline.produce()
    # entirely uncaught, discarding every one of those already-paid clips
    # and surfacing a message that said nothing completed. stitch_error
    # carries the real merge failure reason once that is caught instead,
    # so the owner is told the truth (real clips exist, merging failed,
    # here is why) instead of a false "nothing completed".
    stitch_error: str | None = None
    # Owner's instruction, 2026-10-20: a hard real-money ceiling per run,
    # separate from the cost estimate shown before Approve. Lines never
    # attempted because the ceiling was already reached land here (not
    # failed_lines, which is for a real attempt that failed) so the owner
    # can see exactly which lines were skipped and why, with zero real
    # fal.ai spend against any of them.
    skipped_lines: list[DialogueLine] = field(default_factory=list)
    # Owner's instruction, 2026-10-20: a real reference photo generated
    # per character per screenplay (not a fixed panel reused forever -
    # see Character.appearance) - its real cost must be visible in the
    # same total the owner already checks. Owner's standing instruction,
    # 2026-10-21: a failure to generate one fails that character's own
    # lines (zero cost) rather than falling back to any placeholder -
    # there is none left in this codebase - with the real reason kept
    # here instead of silently swallowed, never discarding the rest of
    # the episode over it.
    portrait_cost_usd: float = 0.0
    portrait_failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.clips) + self.portrait_cost_usd


@dataclass
class PublishResult:
    platform: str
    success: bool
    video_id: str | None = None
    url: str | None = None
    error: str | None = None
