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
    voice: str = "ar-TN-HediNeural"


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

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.clips)


@dataclass
class PublishResult:
    platform: str
    success: bool
    video_id: str | None = None
    url: str | None = None
    error: str | None = None
