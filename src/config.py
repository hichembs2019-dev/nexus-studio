"""Single source of real-vs-mock decisions, mirroring nexus-commerce's own
convention: a missing secret means "use the mock for that piece and keep
going", never a crash.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    groq_api_key: str
    groq_model: str
    gemini_api_key: str
    gemini_model: str
    youtube_client_id: str
    youtube_client_secret: str
    youtube_refresh_token: str
    telegram_bot_token: str
    azure_speech_key: str
    azure_speech_region: str
    azure_tts_lang: str
    # Owner's instruction, 2026-10-21 ("الان اريد البوت بربليكات" - connect
    # the bot to Replicate now), confirmed by a real, owner-run test
    # showing Replicate's billing page matches its quoted price exactly
    # (unlike fal.ai's own real invoice, see docs/SOURCES.md). Owner's
    # later, explicit, repeated instruction the same night ("لا تضع
    # السراق ... احذفهم من الكود" / "خذ وقتك ... هذا ما طلبته منك
    # بالظبط" - remove the thieves [fal.ai] from the code, take your
    # time, this is exactly what I asked): fal.ai is not merely unused
    # by default any more - fal_key/fal_configured/video_provider and
    # their real client files (fal_video_client.py, image_client.py) are
    # deleted outright. Replicate is the only real video/image provider.
    replicate_api_token: str
    # Owner's instruction, 2026-10-22 ("نعم" to continuing the wiring,
    # after the self-hosted Wan 2.2 S2V graph was proven end to end on a
    # real pod - see docs/SOURCES.md): the self-hosted engine is a real,
    # second `ReplicateVideoClient` implementation
    # (self_hosted_video_client.py), preferred over Replicate when both
    # are configured (same priority-by-configuration pattern app.py
    # already uses for Groq-over-Gemini, not a reintroduced
    # `VIDEO_PROVIDER` toggle - that selector was deleted outright when
    # fal.ai was removed, see docs/SOURCES.md, and is not brought back
    # here). Needs a real Runpod API key AND a real, already-provisioned
    # pod id - there is no "create a pod on demand" path, since the real
    # model weights only exist on one specific pod's network volume.
    runpod_api_key: str
    runpod_s2v_pod_id: str
    # Real, live-read rate for the actual production pod (`wt5ennaph7w5nh`,
    # Runpod MCP list-pods, 2026-10-22: one NVIDIA A40, $0.59/hr) - kept
    # configurable, never hard-coded past this one default, because a
    # pod's real billed rate is tied to its own GPU/host and can change
    # if the pod is ever recreated (see self_hosted_video_client.py's own
    # docstring for how this feeds the shared price_per_second_usd knob).
    runpod_s2v_hourly_rate_usd: float
    # Owner's own explicit standing rule, repeated throughout this
    # project's whole history ("لا تصرف اي مال حقيقي... دون ذكر السعر
    # الدقيق اولا والانتظار لـ"نعم" الصريح" - never spend real money
    # without stating the exact price first and waiting for an explicit
    # yes): MultiTalkVideoClient/the real two-speaker pairing logic in
    # DialoguePipeline are built and tested (docs/SOURCES.md, "MultiTalk
    # two-speaker generation proven real, end to end"), but wiring them
    # in here would change what the NEXT real screenplay the owner
    # approves actually spends money on, automatically, the moment this
    # deploys - without the owner having explicitly turned that specific
    # real behavior change on themselves. False (the default) keeps
    # DialoguePipeline's multitalk_client at None - every line still
    # produces its own isolated single-speaker clip, unchanged. Reuses
    # the SAME pod/credentials as self-hosted S2V (same GPU pod ran both
    # proofs) - no separate Runpod config needed, only this one explicit
    # real switch.
    multitalk_enabled: bool
    output_dir: str
    # Owner's instruction, 2026-10-20 ("قلت لك ضع له كود يشتغل و يسحب
    # المال الا عند صنع فيديو" - I told you to put code in that only
    # draws money when a video is actually made): fal.ai itself already
    # only bills for real completed output (confirmed live: a job that
    # times out costs $0.000, see docs/SOURCES.md "fal.ai job timeout").
    # This is a SEPARATE, additional guard: a hard ceiling on total real
    # spend within one production run, so a long or malformed screenplay
    # can never silently run the real bill past what the owner is willing
    # to lose in one sitting, independent of the cost estimate already
    # shown before Approve.
    max_episode_cost_usd: float
    # Cinematic shot mode (2026-10-23): Gemini directs one shot per line,
    # Groq compiles its prompts, flux-2-dev renders a still from the
    # character's portrait, and that still - not the portrait - is
    # animated. Adds one real paid image per spoken line, so it is an
    # explicit owner switch (env CINEMATIC_SHOTS_ENABLED), off by default,
    # same posture as multitalk_enabled: deploying this changes nothing
    # about what the next approved screenplay spends until it is turned on.
    cinematic_shots_enabled: bool = False
    # Kling motion + lip sync (owner's choice, 2026-10-25): preferred over
    # both Wan S2V paths when on. Off by default - deploying this code
    # changes nothing until the owner sets KLING_VIDEO_ENABLED=true AND a
    # KLING_API_KEY from their own Kling account. Unit price defaults to
    # the 180-day pack rate (kling.ai/dev/pricing, 2026-10-25); the trial
    # pack is 0.098.
    kling_api_key: str = ""
    kling_video_enabled: bool = False
    kling_unit_price_usd: float = 0.14
    kling_resolution: str = "720p"

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def youtube_configured(self) -> bool:
        return bool(self.youtube_client_id and self.youtube_client_secret and self.youtube_refresh_token)

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token)

    @property
    def azure_speech_configured(self) -> bool:
        return bool(self.azure_speech_key and self.azure_speech_region)

    @property
    def replicate_configured(self) -> bool:
        return bool(self.replicate_api_token)

    @property
    def self_hosted_s2v_configured(self) -> bool:
        return bool(self.runpod_api_key and self.runpod_s2v_pod_id)

    @property
    def cinematic_shots_configured(self) -> bool:
        return self.cinematic_shots_enabled and self.replicate_configured and (
            self.gemini_configured or self.groq_configured
        )

    @property
    def kling_configured(self) -> bool:
        return self.kling_video_enabled and bool(self.kling_api_key)

    @property
    def multitalk_configured(self) -> bool:
        return self.multitalk_enabled and self.self_hosted_s2v_configured

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            gemini_model=os.environ.get("GEMINI_MODEL", "gemma-4-26b-a4b-it"),
            youtube_client_id=os.environ.get("YOUTUBE_CLIENT_ID", ""),
            youtube_client_secret=os.environ.get("YOUTUBE_CLIENT_SECRET", ""),
            youtube_refresh_token=os.environ.get("YOUTUBE_REFRESH_TOKEN", ""),
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            azure_speech_key=os.environ.get("AZURE_SPEECH_KEY", ""),
            azure_speech_region=os.environ.get("AZURE_SPEECH_REGION", ""),
            # Owner's instruction, 2026-10-19: Syrian ("neutral/white")
            # Arabic, widely understood across North Africa, the same
            # dialect used dubbing Turkish series - what the real
            # DialoguePipeline path actually uses (see character_photos.py).
            azure_tts_lang=os.environ.get("AZURE_TTS_LANG", "ar-SY"),
            replicate_api_token=os.environ.get("REPLICATE_API_TOKEN", ""),
            runpod_api_key=os.environ.get("RUNPOD_API_KEY", ""),
            runpod_s2v_pod_id=os.environ.get("RUNPOD_S2V_POD_ID", ""),
            runpod_s2v_hourly_rate_usd=float(os.environ.get("RUNPOD_S2V_HOURLY_RATE_USD", "0.59")),
            multitalk_enabled=os.environ.get("MULTITALK_ENABLED", "").strip().lower() in ("1", "true", "yes"),
            output_dir=os.environ.get("OUTPUT_DIR", "./output"),
            max_episode_cost_usd=float(os.environ.get("MAX_EPISODE_COST_USD", "10.0")),
            cinematic_shots_enabled=os.environ.get("CINEMATIC_SHOTS_ENABLED", "").strip().lower() in ("1", "true", "yes"),
            kling_api_key=os.environ.get("KLING_API_KEY", ""),
            kling_video_enabled=os.environ.get("KLING_VIDEO_ENABLED", "").strip().lower() in ("1", "true", "yes"),
            kling_unit_price_usd=float(os.environ.get("KLING_UNIT_PRICE_USD", "0.14")),
            kling_resolution=os.environ.get("KLING_RESOLUTION", "720p"),
        )
