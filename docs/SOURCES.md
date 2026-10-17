# Sources - every real fact this code relies on

Same convention as nexus-commerce: every external fact is verified live before
code depends on it, and the evidence stays here so it can be checked again
later without re-guessing.

## Wan 2.2 TI2V-5B real generation (2026-10-17)

- Model identified from a real Hugging Face API read:
  `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` (947 likes, 7.4M downloads, tag
  `image-to-video`, finetune of `Wan-AI/Wan2.2-TI2V-5B`).
- Real file sizes confirmed via `curl -sIL` following the full redirect
  chain (the first hop's `content-length` is the redirect page, not the
  file):
  - `wan2.2_ti2v_5B_fp16.safetensors`: 9,999,658,848 bytes
  - `umt5_xxl_fp16.safetensors`: 11,366,399,385 bytes
  - `wan2.2_vae.safetensors`: 1,409,400,960 bytes
- All three downloaded onto the real network volume and verified
  byte-for-byte against the sizes above (`stat -c%s`).
- ComfyUI's own `/object_info` (1071 real node types) confirmed native
  support (`Wan22ImageToVideoLatent`, `WanImageToVideo`, ...) - no custom
  node install needed.
- `Wan22ImageToVideoLatent`'s `start_image` input is optional (confirmed
  via `/object_info/Wan22ImageToVideoLatent` before dropping it) - this
  project runs pure text-to-video.
- The official Comfy-Org workflow template
  (`Comfy-Org/workflow_templates/templates/video_wan2_2_5B_ti2v.json`)
  names `umt5_xxl_fp8_e4m3fn_scaled.safetensors` for the CLIPLoader; this
  project uses `umt5_xxl_fp16.safetensors` instead - the full-precision
  file actually downloaded, same text encoder, same repo. Deliberate
  substitution, not a guess.
- One real generation submitted via `POST /prompt` (node_errors: {}), its
  progress watched live via `GET /queue`, and its result confirmed via
  `GET /history/<prompt_id>`:
  - `execution_start`: 1792226892192 ms
  - `execution_success`: 1792227095065 ms
  - **real elapsed: 202.873s** for 121 frames @ 1280x704, 20 steps,
    `uni_pc`/`simple`, on a real RTX 4090 (RunPod Secure Cloud, $0.89/hr).
  - Output `nexus_studio_test_00001_.mp4` downloaded via `GET /view` and
    confirmed a real, valid MP4 (`file` reports "ISO Media, MP4 Base Media
    v1"), 563,428 bytes.
- `src/workflows/wan_prompt_builder.py` encodes this exact graph (node ids
  37/38/39/48/6/7/55/3/8/57/58 kept identical to the real run).

Economics derived from the above, not measured separately: 121 frames /
24fps = 5.0417s of output per 202.873s of GPU time → **~89.5s (~1.5 min)
of finished video per GPU-hour** at this resolution/step count. This is
raw-generation cost only - no script writing, multi-shot coverage, retries,
or editing.

## ComfyUI's output directory (2026-10-17)

`find /workspace/runpod-slim/ComfyUI -maxdepth 1 -iname output -type d`
confirmed `/workspace/runpod-slim/ComfyUI/output` on the real pod, and the
test clip above was found there at exactly 563,428 bytes - matching the
`/view` download byte for byte. `src/workflows/episode_pipeline.py`'s
`COMFYUI_OUTPUT_DIR` constant is this verified path.

## SSH blocked, Jupyter remote-exec works (2026-10-17)

- A real `ssh -i <registered-key> ...@ssh.runpod.io` connection timed out
  ("Connection timed out") from this project's own execution sandbox.
- The sandbox's own proxy status endpoint
  (`curl $HTTPS_PROXY/__agentproxy/status`) confirmed only HTTP/HTTPS
  traffic is proxied - there is no raw TCP (port 22) path out, regardless
  of whether RunPod's side is configured correctly. This is a sandbox
  limitation, not a RunPod one.
- Workaround proven live: the pod's Jupyter server (port 8888/http) goes
  through the same HTTPS proxy. Token login
  (`GET /login?token=<JUPYTER_PASSWORD>`) plus a follow-up `GET /tree` to
  pick up the `_xsrf` cookie, then `POST /api/kernels` with
  `X-XSRFToken` header starts a real kernel; a websocket to
  `/api/kernels/<id>/channels` with the session cookies, `X-XSRFToken`
  header, and an `Origin` header matching the Jupyter host executes real
  code and streams its stdout back. A real `nvidia-smi` call through this
  path printed the real GPU: `NVIDIA GeForce RTX 4090, 24564 MiB, 396 MiB
  used`.
- Two auth pitfalls found only by testing, both now locked in by
  `tests/test_remote_exec_client.py`:
  - the websocket upgrade 403s (Jupyter's own HTML 403 page, not a JSON
    error) without the `Origin` header, even with valid cookies and a
    valid `X-XSRFToken`.
  - a hand-parsed Netscape cookie jar must treat a `#HttpOnly_`-prefixed
    line as a real cookie, not a comment - skipping every `#`-prefixed
    line silently drops the session identity cookie. (This project avoids
    the pitfall entirely by using `httpx.Client`'s own cookie jar instead
    of hand-parsing a cookie file, but the lesson is why that choice was
    made.)
- Real ffmpeg confirmed present on the pod (`ffmpeg version 6.1.1-3ubuntu5`,
  via the same remote-exec path) - this is what
  `EpisodePipeline._stitch_remote` runs for multi-scene episodes, so clips
  never need to round-trip through this process to be concatenated.

## Groq agent reasoning

Same real endpoint and request/response shape verified live for
nexus-commerce on 2026-10-14 (see that project's `docs/SOURCES.md` "Groq
fallback"): `POST https://api.groq.com/openai/v1/chat/completions`,
`Authorization: Bearer <key>`, OpenAI-compatible
`choices[0].message.content`. Reused here as an already-verified fact.

## Groq org-wide quota (2026-10-17)

First real end-to-end test of `WriterAgent`/`DirectorAgent.plan_episode_from_screenplay`
used nexus-commerce's existing Groq key and hit a real `429`-shaped error
on the Writer call:

```
Rate limit reached for model `openai/gpt-oss-120b` in organization
org_01m4xav43be8j956tk3w9jt3tg ... tokens per day (TPD): Limit 200000,
Used 199562
```

A second, brand-new Groq key was created (named "Nexora studio" in the
Groq console, confirmed via screenshot alongside the existing "Nexora
commerce" key - both listed under the same `Personal / Default Project`).
Retrying with the new key hit the exact same wall seconds later (`Used:
199974/200000`, same `organization` id) - **a second key under the same
Groq account does not get its own TPD quota**; the limit is scoped to the
account/organization, not the key. This directly invalidated an earlier
assumption in this project's own commit history ("a dedicated Groq key
avoids quota contention with nexus-commerce") - corrected here rather
than left standing once the live test disproved it.

Fix: `RealGroqReasoningClient` gained the same optional `fallback`
constructor parameter nexus-commerce's own client already uses, wired in
`src/app.py` to `RealGeminiReasoningClient` (a genuinely separate
account/quota - Google AI, not Groq) when `GEMINI_API_KEY` is set. A real
end-to-end run (Writer screenplay → Director visual prompts) through this
wiring succeeded end to end the same day, producing a coherent Arabic
screenplay and its correctly-translated, dialogue-free English visual
prompts.

## YouTube Data API v3 upload (2026-10-17)

Verified from Google's own official docs (`developers.google.com`), not a
live authenticated call yet - see "Publishing - platforms not yet built"
below for exactly what's still missing.

- **Resumable upload, step 1** (start a session):
  `POST https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status`,
  headers `Authorization: Bearer <access_token>`,
  `Content-Type: application/json; charset=UTF-8`,
  `X-Upload-Content-Length: <file size>`, `X-Upload-Content-Type: video/*`
  (this project sends `video/mp4`, matching the real `.mp4` files
  `EpisodePipeline` produces). Body: a `video` resource JSON
  (`snippet.title`/`description`/`tags`, `status.privacyStatus`).
  Response: `200 OK`, empty body, the session URI in the `Location`
  header.
- **Resumable upload, step 2** (the bytes): `PUT` to that exact URI,
  `Content-Length`/`Content-Type` matching step 1's `X-Upload-*` headers,
  raw video bytes as the body. Success: `201 Created`, a `video` resource
  JSON body whose `id` field is the new video's id.
- **OAuth token refresh**:
  `POST https://oauth2.googleapis.com/token`,
  `application/x-www-form-urlencoded` body
  `{client_id, client_secret, refresh_token, grant_type: "refresh_token"}`
  (the page states this value literally). Response:
  `{access_token, expires_in, scope, token_type}`.
- `src/integrations/publishing_client.py`'s `RealYouTubeClient` encodes
  exactly this - `tests/test_publishing_client.py` pins the real header
  names, the `grant_type` value, and the two-step flow (one test was
  mutation-tested: flipping `grant_type` to `authorization_code` fails the
  test, confirming it actually checks the value rather than just
  existing).

## Publishing - platforms not yet built

Only YouTube has even an unverified-but-doc-sourced client. Nothing exists
yet for the others - guessing a request shape nobody has read the docs
for is exactly what this project has committed not to do. What each one
would actually need, so the next session doesn't have to rediscover it:

- **YouTube** (closest to real): a Google Cloud project with the YouTube
  Data API v3 enabled, an OAuth 2.0 client (client_id/client_secret), and
  one real browser consent flow authorizing the
  `https://www.googleapis.com/auth/youtube.upload` scope to get a
  `refresh_token` - `RealYouTubeClient` only does the refresh→access_token
  exchange after that, never the initial consent screen. None of this
  exists yet; `YOUTUBE_CLIENT_ID`/`YOUTUBE_CLIENT_SECRET`/
  `YOUTUBE_REFRESH_TOKEN` are all unset, so `build_app` leaves `"youtube"`
  out of the publishing registry entirely (see `src/app.py`).
- **TikTok**: Content Posting API requires an approved developer app
  (manual review by TikTok, not instant) before any upload call works at
  all - docs not yet read here.
- **X (Twitter)**: video upload via API v2 needs a paid API tier for
  reasonable size/duration on the free tier - docs not yet read, pricing
  not yet re-confirmed for 2026.
- **Instagram**: Graph API via a Meta/Facebook app plus a connected
  Instagram Business account and its own access token - docs not yet
  read.
- **Snapchat, Twitch, Kick**: not researched at all yet.
