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
