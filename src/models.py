"""Episode planning shapes - the Director Agent's output, consumed by the
generation pipeline that turns each Scene into one real Wan 2.2 clip.

CLIP_SECONDS is fixed by the proven workflow itself, not a free choice: one
Wan 2.2 TI2V-5B generation call (121 frames at 24fps) produced a real,
playable 5.04s clip in 202.87s on an RTX 4090 - verified live 2026-10-17,
see docs/SOURCES.md "Wan 2.2 TI2V-5B real generation". An episode's scene
count is therefore however many 5.04s clips its target length needs.

GPU_HOURLY_COST_USD is the real RunPod Secure Cloud RTX 4090 price quoted
live the same day (see docs/SOURCES.md "RunPod GPU pod") - used only to
report real cost, never to gate generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CLIP_FRAMES = 121
CLIP_FPS = 24
CLIP_SECONDS = CLIP_FRAMES / CLIP_FPS
GPU_HOURLY_COST_USD = 0.89


@dataclass
class ScreenplayScene:
    index: int
    text: str


@dataclass
class Screenplay:
    """The Writer Agent's output: a human-readable draft the owner reads
    and edits directly (ScreenplayScene.text) before anything is ever
    generated. Nothing downstream runs on a Screenplay until the owner
    hands the edited version to DirectorAgent.plan_episode_from_screenplay
    - there is no automatic chaining from Writer to Director."""

    title: str
    premise: str
    scenes: list[ScreenplayScene] = field(default_factory=list)


@dataclass
class Scene:
    index: int
    positive_prompt: str
    negative_prompt: str
    seed: int


@dataclass
class Episode:
    title: str
    premise: str
    scenes: list[Scene] = field(default_factory=list)

    @property
    def target_seconds(self) -> float:
        return len(self.scenes) * CLIP_SECONDS


@dataclass
class ProducedClip:
    scene: Scene
    local_path: str
    generation_seconds: float


@dataclass
class ProducedEpisode:
    episode: Episode
    clips: list[ProducedClip]
    stitched_path: str | None
    total_generation_seconds: float

    @property
    def total_cost_usd(self) -> float:
        return self.total_generation_seconds * (GPU_HOURLY_COST_USD / 3600)


@dataclass
class Character:
    """A speaking role, carried across an episode (and meant to be reused
    across a whole series) by a FIXED reference photo - Wan 2.2's
    speech-to-video call (see fal_video_client.py) takes one reference
    image per clip, and reusing the same image for every line this
    character speaks is what keeps their face the same person scene to
    scene. Without a real photo here there is no consistency to keep -
    DialoguePipeline raises rather than silently generating a different
    face per line."""

    name: str
    reference_image_path: str
    voice: str = "ar-SY-LaithNeural"
    # Owner's instruction, 2026-10-20, rejecting a fixed panel of six
    # generic photos reused across every story forever ("لا تكون صور
    # ثابتة الى الابد ... بل حسب القصة و السيناريو" - not fixed forever,
    # but according to the story and screenplay): a short visual
    # description (hair, build, distinguishing features) inferred from
    # THIS premise by the same cast-writing LLM call that already infers
    # gender/age_group - DialoguePipeline uses it to generate a real,
    # story-specific reference photo per character instead of reusing one
    # of a handful of generic ones across unrelated stories.
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
    # same total the owner already checks, and a failure to generate one
    # must fall back to the cheap generic placeholder for that character
    # rather than aborting the whole episode, with the real reason kept
    # here instead of silently swallowed.
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
