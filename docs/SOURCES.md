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

## YouTube Data API v3 upload (2026-10-17, live-verified)

Real wire shapes from Google's own official docs, then proven with real
authenticated calls the same day against a real channel (see "Nexora
channel setup" below for how the credentials themselves were obtained).

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

## Nexora channel setup (2026-10-17)

Full real setup, done once, end to end, no guessing - kept here so it
never has to be rediscovered.

**Google Cloud / OAuth client:**
1. A Google Cloud project, "YouTube Data API v3" enabled.
2. OAuth consent screen: User type **External** (the only option for a
   personal, non-Workspace Google account), app name, support/contact
   email. Stayed in **Testing** status - no Google review needed since
   only the owner's own account uses it. The owner's own email was added
   under **Test users** - without this, the consent screen rejects the
   login entirely.
3. OAuth client ID, type **Web application**, one authorized redirect URI:
   `https://developers.google.com/oauthplayground`. The "Use this client
   for an AI-powered agent" checkbox was left OFF - its real effect is
   undocumented here, so it was not enabled on an unverified guess.
4. **Real finding**: the very first resumable-upload call with a brand
   new OAuth client failed with a real `400 Bad Request`, then an
   identical call succeeded minutes later. Google's own UI warns
   "l'applicazione delle impostazioni potrebbe richiedere da cinque minuti
   a qualche ora" (settings can take five minutes to a few hours to
   propagate) - this is that delay, not a bug in the client code.

**Getting a refresh_token (Google OAuth Playground):**
`developers.google.com/oauthplayground` → gear icon → Access type
**Offline**, check "Use your own OAuth credentials", paste the client
ID/secret → scope box (bottom of the API list) → paste
`https://www.googleapis.com/auth/youtube.upload` → Authorize APIs → sign
in as the test user → accept the real "Google hasn't verified this app"
warning (expected for a Testing-status app, not a sign of a problem) →
Exchange authorization code for tokens → `refresh_token` appears in the
JSON response.

**Real finding - transcription errors are not negligible**: reading a
long `refresh_token` off a phone screenshot produced a token that failed
with `invalid_grant` twice (an `O`/`0` and a missing character each time).
Copying the raw page text (long-press → select all → copy) and reading
the token out of that exact text fixed it immediately both times. For any
credential this long, treat a screenshot-read value as provisional until
tested, and prefer copied text over an image whenever the sender can get
it.

**Real finding - a video description can get a real validation error**:
the first real upload attempt with a description containing `->` failed
with
`{"error": {"code": 400, "message": "The request metadata specifies an invalid video description.", "reason": "invalidDescription"}}`.
Isolated empirically (not guessed) by retrying three description strings
differing only in whether they contained `->` - only that one failed.
Avoid literal `->` arrows in descriptions built for this client.

**Real finding - a Brand Account channel needs its own authorization,
and `videos.list`/`videos.delete` need a broader scope**: the channel's
display name ("Nexora", logo "Nexora Global") was set up after the first
real upload, through YouTube Studio directly (Create a channel → custom
name) rather than through this project's code - no Data API v3 endpoint
creates a channel for an ordinary account. A second, separate OAuth
Playground run (fresh authorization code, fresh refresh_token) was needed
afterward; the resulting upload was confirmed to land on the Nexora
channel by checking YouTube Studio's own channel switcher in the browser,
not through this project's code, since the `youtube.upload` scope this
client intentionally requests does **not** cover `videos.list` (confirmed
live: a real `channels`/`videos` read with an upload-scoped token returned
`403 ACCESS_TOKEN_SCOPE_INSUFFICIENT`) or `videos.delete` (its own docs
list only `youtube`, `youtubepartner`, `youtube.force-ssl` as valid
scopes - deliberately broader than this project requests, so cleanup of
stray test uploads was done by hand in YouTube Studio, not via this
client).
- **Real finding - channel renames are rate-limited**: YouTube rejected
  two name values in a row ("Nexora studio" - likely a reserved word
  pattern; a second value that was actually meant for the description
  field, a mistake caught before it cost another attempt) and then
  blocked further rename attempts for **24 hours** with "Hai inserito
  troppi nomi che non possono essere utilizzati" - independent of whether
  the next name tried would itself have been valid. The channel's image
  and description saved without being affected by the name being locked.
- Two real videos were uploaded to the live Nexora channel as proof
  (`private`, since this was setup verification, not real content):
  `https://youtu.be/c5SITEMGKvo` (3.37s upload) and
  `https://youtu.be/3ZcGjjlJo84` (2.96s upload, the one confirmed to
  target Nexora specifically). Both, plus three orphaned pending sessions
  left over from the description-isolation testing above (a `POST` that
  returns `200`+`Location` already creates a real, visible placeholder
  video entry even if step 2 is never sent), were deleted by hand from
  YouTube Studio afterward - this project's `RealYouTubeClient` has no
  delete method, deliberately: it was built with only the `youtube.upload`
  scope it actually needs.

Real values now live in this project's own (gitignored) `.env`:
`YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, `YOUTUBE_REFRESH_TOKEN` -
all pointed at the Nexora channel specifically, not the owner's personal
one.

## Publishing - platforms not yet built

YouTube now has a live, working client against a real channel (see
"Nexora channel setup" above). Nothing exists yet for the others -
guessing a request shape nobody has read the docs for is exactly what
this project has committed not to do. What each one would actually need,
so the next session doesn't have to rediscover it:

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

## Telegram bot interface (2026-10-17)

The owner wanted a persistent way to talk to the agents without going
through this conversation each time - same role Telegram already plays
for nexus-commerce.

- **Reused, already-verified facts** (from nexus-commerce's own
  `src/integrations/telegram_client.py` and `docs/SOURCES.md`
  "Telegram"): the `python-telegram-bot` library, `Bot(token=)`,
  `bot.send_message(chat_id=, text=, reply_markup=)`,
  `InlineKeyboardMarkup`/`InlineKeyboardButton`, and the
  `Application`/`CallbackQueryHandler`/`update.callback_query` shape for
  receiving button presses.
- **Newly verified**: `CommandHandler(command, callback)`'s callback
  signature (`async def callback(update, context)`) and that
  `context.args` holds the command text split on whitespace - confirmed
  from python-telegram-bot's own docs.
- **Not independently doc-verified**: `Bot.send_video`'s exact parameter
  names and `reply_video`. Both WebFetch attempts (python-telegram-bot's
  own docs and Telegram's core Bot API page) truncated before reaching
  the parameter tables - too large to fetch in full for this. The
  `video=<open file>` call shape mirrors `send_message`'s already-verified
  pattern, but treat this one as integration-only until exercised with a
  real bot token, same as nexus-commerce already treats
  `telegram_listener.build_application`.
- **The 50MB-per-video cap for bot uploads** is corroborated by several
  third-party docs mirrors (not fetched from Telegram's own page
  directly, which truncated before that section). Short episodes
  (seconds, not minutes, at this project's current resolution) stay well
  under it - the real stitched test episode was 2.5MB for ~10s of video.

**Design**: `src/bot/conversation.py` (`BotConversation`) holds all the
decision logic - pure, synchronous, fully unit-tested (18 tests) against
Mock agents, no Telegram library involved. `src/bot/runner.py` is the
thin, not-unit-tested async wiring, matching nexus-commerce's own
documented split for exactly this reason (`telegram_listener.py`'s own
docstring). `EpisodePipeline.produce()` takes minutes and is synchronous,
so it runs in the asyncio default executor (a background thread) rather
than blocking the bot's event loop - confirmed by design, not yet by a
real multi-chat concurrent run.

**Commands**: `/new [seconds] <idea>` (default 30s) → Writer drafts a
screenplay, sent back with Approve/Cancel buttons. `/edit <scene#> <new
text>` revises one beat and resends the draft. Approve → Director
converts it to visual prompts → EpisodePipeline generates and stitches →
the finished video is sent back with a Publish-to-YouTube button.

**Real finding, 2026-10-17 - conversation state does not survive a
redeploy**: `BotConversation._states` is a plain in-memory dict. A real
owner session hit this directly: a screenplay was approved, a redeploy
(rolling the pod URL after the original RunPod host ran out of capacity,
see "RunPod host capacity" below) restarted the bot container mid-flow,
and the next Approve tap on the still-visible old message came back "no
draft screenplay" - the state was gone. Known limitation, not yet fixed;
a real-state fix (SQLite, matching nexus-commerce's own persistence
convention) is the natural next step once this project needs to survive
redeploys without losing an in-progress draft.

## RunPod host capacity (2026-10-17)

A stopped pod's GPU is NOT reserved for it - starting it again can fail
with a real `400`:
`"There are not enough free GPUs on the host machine to start this pod."`
(hint: "For a stopped Pod, its host's GPUs are taken; that Pod can only
start once they free up.") Confirmed live: the original pod
(`6enpyl3ark8ck8`) failed to restart after being stopped to save money:
`get-data-center(EU-RO-1, include: GPU_AVAILABILITY)` showed real RTX
4090 stock at "LOW" (not zero) for the region, but that specific host had
none free. The real fix was not retrying the same pod but creating a
*new* pod (`create-pod`, same template/GPU/region, same real cost
$0.89/hr) with the **same network volume** (`hwee6i9g4o`) mounted - a new
pod can land on a different host with free capacity, and the volume
(with every model already on it) is independent of which pod mounts it.
The dead pod was terminated afterward. Net effect: a region-level "stop
to save money, start again later" policy is not actually safe for a
single pinned pod on scarce stock - expect to recreate the pod (reusing
the volume) some fraction of the time, and update
`COMFYUI_BASE_URL`/`JUPYTER_BASE_URL`/`JUPYTER_PASSWORD` (pod-id-specific)
wherever they're configured (this project's `.env`) when it happens.

## Wan 2.2 real content gaps (2026-10-17)

Both found live, by the owner, from actual output - not caught by any
test, because neither is a code bug:

- **No audio at all.** Wan 2.2 TI2V-5B (as used here, pure text-to-video)
  produces silent video - this was never built or promised, and should
  have been stated plainly before the first real test rather than left
  for the owner to discover. Real dialogue/voice would need a separate
  pipeline stage (TTS + lip-sync, or a different Wan 2.2 variant such as
  the S2V - speech-to-video - checkpoints already listed in the
  Comfy-Org repo's file manifest, see "Wan 2.2 TI2V-5B real generation"
  above) - not built, real additional GPU time and engineering.
- **Wrong ethnicity by default.** A screenplay about a Tunisian family,
  translated into English-only visual prompts with no ethnicity stated,
  rendered East Asian faces - Wan 2.2's own default bias when appearance
  isn't specified (a Chinese-developed model). Fixed in
  `DirectorAgent`'s screenplay-to-visual system prompt: when the
  screenplay's names/language/setting imply a specific ethnicity or
  nationality, every scene description showing a face must state it
  explicitly, consistently across all scenes of the same characters -
  locked in by
  `test_plan_episode_from_screenplay_system_prompt_instructs_carrying_ethnicity`.
  Not yet re-verified with a real generation (the fix was made but no
  new real video has been produced with it yet).
