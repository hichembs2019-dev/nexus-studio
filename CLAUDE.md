# nexus-studio

## Owner's standing rule — never state anything as true without real proof

Same standing instruction as nexus-commerce (owner's own words,
2026-10-18):

> يجب عليك الا تكذب في المستقبل و الحاضر / لا تذكر شيء الا انت متاكد و
> تعرفه حقيقي و بدون تخمين او افتراض او كذب
> (Never lie, now or in the future. State nothing unless you are certain
> and know it is real — no guessing, no assuming, no lying.)

What this means in practice here:

- **A video is "produced" only once a real mp4 file exists on disk and
  was inspected** (ffprobe duration, has_audio, real byte size) — not
  once the pipeline code runs without raising. The real 2026-10-18 run
  of `DialoguePipeline` reached fal.ai successfully and still produced
  **zero real clips** (403 Exhausted balance) — see docs/SOURCES.md
  "Real character dialogue with lip sync". That is the standard: report
  the real HTTP status and the real account state, not "the pipeline
  works."
- **96 passing tests prove the code's logic, not a finished product.**
  State what is actually wired end to end (direct Python call today, not
  the Telegram bot) and what is not, every time status is reported.
- **If context was cleared and you can't see an earlier claim, say
  exactly that** rather than defending or denying it from memory.
