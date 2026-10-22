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

## Azure TTS voiceover (2026-10-17)

Real fix for the first gap above ("no audio at all"). Everything below was
checked live against a real, owner-created Azure resource - not assumed.

### Resource

Created via the real, current official path for this: `portal.azure.com`
→ search "Microsoft Foundry" → "Creare una risorsa Foundry". The
"Earlier-generation"/classic Speech resource blade was the first thing
tried and was wrong - corrected after fetching Microsoft's own quickstart
docs rather than guessing a second time.

- Resource name: `nexus-studio-speech`, resource group `nexus-studio-speech`
- Region: `eastus` - chosen because it was the portal's own default, then
  verified (not assumed) against the real region/feature table at
  `learn.microsoft.com/azure/ai-services/speech-service/regions`: `eastus`
  shows ✅ for Neural text to speech, HD voices, Personal voice and Voice
  conversion.
- Endpoint: `https://nexus-studio-speech.services.ai.azure.com/`
- Real key/region live in the server's `.env` as `AZURE_SPEECH_KEY` /
  `AZURE_SPEECH_REGION=eastus`, never committed (`.env` is gitignored).

### The real REST contract (classic `cognitiveservices/v1`, not Voice Live)

Verified against Microsoft's own docs, then proven against the real
resource with two live calls the same day:

- `GET https://eastus.tts.speech.microsoft.com/cognitiveservices/voices/list`
  with only `Ocp-Apim-Subscription-Key` → real `200`, 958 voices returned.
  Filtered for `ar-TN`: **`ar-TN-ReemNeural` (female) and `ar-TN-HediNeural`
  (male), both `Status: GA`** - voice hosting is NOT limited to a handful
  of regions as the general region/feature table alone might suggest; this
  was checked directly against the real resource rather than assumed from
  that table.
- `POST https://eastus.tts.speech.microsoft.com/cognitiveservices/v1` with
  `Ocp-Apim-Subscription-Key`, `Content-Type: application/ssml+xml;
  charset=utf-8`, `X-Microsoft-OutputFormat: audio-24khz-48kbitrate-mono-mp3`,
  `User-Agent`, and an SSML body (`<speak version='1.0' xml:lang='ar-TN'>
  <voice name='ar-TN-HediNeural'>...</voice></speak>`) → real `200`, a
  genuine 25,920-byte, 4.32s playable MP3 (`ffprobe` confirmed: MPEG
  Layer III, 48kbps, 24kHz, mono). **No token exchange step is needed** -
  the subscription key works directly on the synthesize call itself.
  Sent to the owner as a real file so they could hear it directly on the
  phone, per the "has to be visible/audible on the device itself" rule.

### Pipeline wiring (`src/workflows/episode_pipeline.py`)

`EpisodePipeline` takes an optional `TTSClient`; `produce()` takes an
optional `narration_texts` list aligned 1:1 with `episode.scenes` by
index. The narration source is the **Screenplay's own scene text** - the
original-language narrative beat the owner wrote/edited - not the
Director's English visual prompt, which describes the shot, not what
should be said. `BotConversation.handle_approve()` now returns
`(reply, episode, narration_texts)` instead of `(reply, episode)`; wired
through `runner.py`'s `_produce_and_notify` into `pipeline.produce()`.

Two real defects found and fixed before this shipped, each with a
mutation-tested guard (removed, confirmed the test failed; restored,
confirmed it passed again):

1. **A longer narration must not be cut to the clip's length.** A plain
   audio/video merge (`ffmpeg -i video -i audio -c:v copy -c:a aac`, no
   `-shortest`) does NOT truncate the *container* - `ffprobe` reports the
   full audio-length duration regardless of whether the video stream was
   actually extended. The first version of this guard checked container
   duration and passed even with the real fix deleted - a false guard,
   the same category of mistake this project has hit before (scene
   watchdog, screen wake tests). Rewritten to check the **video stream's
   own frame count** instead
   (`test_narration_longer_than_the_clip_freezes_the_last_frame_instead_of_cutting_it_short`),
   which does fail without the fix. The real fix: when `audio_duration >
   video_duration`, `ffmpeg`'s `tpad=stop_mode=clone` filter freezes the
   last real video frame to cover the rest, so the full narration is
   always heard.
2. **Multi-scene stitching must not silently drop the audio.** The
   existing remote stitcher (`_stitch_remote`) concatenates the
   *original silent* clips still sitting in ComfyUI's own output
   directory on the GPU pod - it has no idea the narrated copies exist
   locally. Left unchanged, a multi-scene episode with narration would
   stitch back together perfectly and be completely silent again at the
   very last step. Fixed: when any scene was narrated, `_finalize()`
   stitches the local narrated clips instead
   (`_stitch_local`, concat demuxer, re-encoded for stream consistency)
   and never calls the remote stitcher - guarded by
   `test_multi_scene_with_narration_stitches_locally_and_keeps_the_audio`.

`ffmpeg`/`ffprobe` are required for all of this and are now installed in
the Docker image (`Dockerfile`: `apt-get install ffmpeg` before the pip
install step) - without them every episode goes back to silent even with
`AZURE_SPEECH_KEY` set.

### Not yet done

- No real end-to-end test has been run yet with the deployed bot (real
  Groq/Director screenplay → real GPU clip → real Azure narration →
  real stitched output with audio, heard on the owner's phone). The unit
  tests above prove the merge/stitch logic is correct against real
  ffmpeg; they do not replace a real production run.
- Narration length vs. `CLIP_SECONDS` (5.04s) trade-off is unmeasured in
  practice: a long screenplay beat read by Azure TTS may regularly run
  well past 5s, which means regularly-frozen tails rather than natural
  motion for most of the narration. If that turns out to be common, the
  real fix is sizing each scene's `length` (Wan prompt builder, already a
  free parameter) to the narration's own measured duration instead of a
  fixed 121 frames - not built, would need the narration synthesized
  *before* the Director/generation step rather than after.

## Real character dialogue with lip sync (2026-10-18)

**Why this exists**: the narration approach above (a TTS voice reading
the screenplay's narrative text over a silent T2V clip) is NOT what was
asked for. The real ask: "وكلاء لصنع المسلسلات و الافلام ... لربح المال"
- agents that make series and films with actual characters speaking and
acting, to publish and monetize - a narrated documentary over silent
clips is a structurally different product. This section is the real,
live-verified replacement path: real character dialogue, lip-synced to
real audio, not a narrator.

### fal.ai Wan 2.2 Speech-to-Video real contract

Checked directly against fal's own docs and model pages (not
third-party price aggregators, which gave conflicting numbers from
$0.015 to $1.00 per video for "Wan 2.2" without saying which variant):

- Endpoint: `fal-ai/wan/v2.2-14b/speech-to-video`. Required inputs
  (confirmed on the model's own page): `prompt` (text), `image_url`
  (reference photo - this is what keeps a character's face the same
  across every line/scene), `audio_url` (the line's own speech audio).
  Both URL fields accept a plain `data:` URI, so a local reference photo
  and a local TTS clip can be sent directly with no separate upload step.
- Real pricing, from the model's own page: **$0.10/s at 480p, $0.15/s at
  580p, $0.20/s at 720p**, video seconds counted at a fixed 16fps
  regardless of the real output fps.
- Queue REST contract (fal's own docs,
  `fal.ai/docs/model-apis/model-endpoints/queue`): `POST
  https://queue.fal.run/<endpoint>` with `Authorization: Key $FAL_KEY`
  returns `{request_id, status_url, response_url, cancel_url}`; poll
  `GET <status_url>` until `status: COMPLETED`; `GET <response_url>`
  returns the model's own output shape, `{"video": {"url": ...}}` for
  this model - a second real GET on that url is the actual mp4 bytes.
  All implemented as-is in `src/integrations/fal_video_client.py`
  (`RealFalVideoClient`), no guessed field names.

### Real end-to-end run, 2026-10-18 - blocked on billing, not on code

Ran the real pipeline (`DialoguePipeline.produce()`, the actual
production code path, not a standalone script) with a real
`FAL_KEY`, a real `AZURE_SPEECH_KEY`, one real character, and one real
Arabic line ("العشاء جاهز يا جويدة"). The submit request reached
fal.ai and was accepted in shape - no validation error - but came back:

```
403 {"detail":"User is locked. Reason: Exhausted balance. Top up your
balance at fal.ai/dashboard/billing."}
```

This matches the $0.00 balance visible in the owner's own fal.ai
dashboard screenshot the same day. **No real clip has been produced
yet** - this is an account billing gap, not a code defect, and is
reported as exactly that rather than claimed as a working pipeline. The
real next step is the owner adding real funds/a payment method at
`fal.ai/dashboard/billing`; the code path itself needs no further
change to produce a real result once that is done.

## Placeholder character photos/voices, auto-assigned (2026-10-19)

Owner's instruction ("الصور طبعا افتراضية... مجموعة صور افتراضية جاهزة
(رجل/امرأة بأعمار مختلفة)، تُسند تلقائيًا حسب وصف كل شخصية" - the photos
are of course placeholders, a ready set of placeholder photos (man/woman,
different ages), assigned automatically per character description): this
closes the "Character reference photos are not auto-generated" gap this
file used to list.

- Six original, Pillow-generated illustrations
  (`assets/placeholder_characters/{male,female}_{young,middle,older}.png`,
  `src/workflows/generate_placeholder_avatars.py`) - not real people, so
  there is no likeness/copyright question to verify, unlike a real-person
  stock photo would raise.
- `DialogueWriterAgent`'s cast-step prompt now also asks for each
  character's `gender`/`age_group`; `src/workflows/character_photos.py`
  resolves those to one of the six photos and to
  `ar-TN-HediNeural`/`ar-TN-ReemNeural` (male/female - both real,
  live-verified Tunisian Arabic Azure voices, see "Azure TTS voiceover"
  above). Every `Character` DialogueWriterAgent builds now carries a real
  `reference_image_path` and `voice` - never the empty string
  `DialoguePipeline.produce()` refuses to accept.
- Covered by `tests/test_dialogue_writer.py::test_write_dialogue_screenplay_assigns_different_photos_by_gender_and_age`
  (distinct photo/voice for a male-older vs. female-young character) and
  `::test_write_dialogue_screenplay_raises_if_cast_item_is_missing_a_required_field`.

## Bot rewired to the real dialogue pipeline (2026-10-19)

Owner's instruction ("بل اطلب من البوت مثلما كان يفعل يرسل لي السيناريو و
يسالني الموافقة" - have the bot do what it used to: send me the
screenplay and ask for approval): this closes the "DialogueWriterAgent/
DialoguePipeline are not wired into the Telegram bot" gap. `/new` now
builds a `DialogueScreenplay` via `DialogueWriterAgent` (line count
estimated from the requested duration, `ESTIMATED_SECONDS_PER_LINE = 4.0`
in `src/bot/conversation.py`) and sends it with the same Approve/Cancel
buttons the old flow used; Approve runs `DialoguePipeline.produce()` in
the background executor (`src/bot/runner.py`) - the one real-money moment
(fal.ai speech-to-video spend), gated behind that same button, same as
before. `App.conversation`/`App.dialogue_pipeline` are `None` together
when `FAL_KEY`/`AZURE_SPEECH_KEY` are not configured; `src/main.py` now
checks for that and exits with a clear log line instead of starting a bot
with no pipeline to drive. The old `WriterAgent`/`DirectorAgent`/
`EpisodePipeline` narration path still exists in the codebase (reachable
directly, as README.md's "direct run" path already documented) but is no
longer what the live bot runs.

**Real bug found by the rewrite's own test, not by inspection**:
`tests/test_bot_conversation.py::test_publish_youtube_success_reports_the_real_url`
failed with `'ProducedDialogueEpisode' object has no attribute 'episode'`
the first time it ran - `PublishingAgent.publish_episode` read
`produced.episode.title`/`.premise` unconditionally, a field that exists
on the old `ProducedEpisode` but not on `ProducedDialogueEpisode` (whose
title/premise live on `.screenplay`). Publishing a real finished dialogue
episode would have raised that `AttributeError` inside the `try`, which
the method's own per-platform fault tolerance would have swallowed into a
silent `"No client configured"`-shaped failure report rather than
crashing visibly - the ✅ branch was dead code for every dialogue episode
until fixed. Fixed in `src/agents/publisher.py` with a small
`_title_and_premise()` helper that checks which of `.episode`/
`.screenplay` the produced object actually has. Mutation-tested by
construction: the test above is the one that caught the bug before the
fix existed, and fails again if the `isinstance` branch is removed.

## Real deployment, 2026-10-19 - a second real bug found live on first use

Deployed to the owner's existing DigitalOcean droplet (`ubuntu-s-1vcpu-512mb-10gb-fral1`, IP `159.89.22.193` - already running `nexus-commerce` in the same `docker-compose.yml`; this was already real, paid infrastructure, not newly created). `git pull` + `docker compose up -d --build nexus-studio`, real `FAL_KEY` added to `.env` from the owner's own fal.ai dashboard. Confirmed live from the container's own logs: `nexus-studio bot starting (long-polling)`, real Telegram `getMe`/`deleteWebhook` calls both `200 OK`, `Application started`.

**First real `/new` sent by the owner hit a genuine bug**, caught from the real logs, not inspection:

```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
...
DialogueWriterError: Cast reply was not valid JSON: ''
```

Root cause: `RealGroqReasoningClient.complete()` only treated an explicit `"error"` key in Groq's response as a failure. A real response came back `200 OK` with **no `"error"` key at all, but a genuinely empty `choices[0].message.content`** - this was returned as `""`, a seemingly-successful empty reply, so the `fallback` to `RealGeminiReasoningClient` (a real, separate-quota account, already wired and configured) was never attempted. `DialogueWriterAgent` then failed four calls later trying to JSON-parse `''`, with an error message that gave no hint the real cause was upstream in the reasoning call. Fixed: an empty/whitespace-only `content` is now treated exactly like an `"error"` key - it raises, which either reaches the real Gemini fallback or surfaces a clear `"empty content"` message instead of a confusing downstream JSON error. Mutation-tested: `test_groq_complete_falls_back_on_a_real_200_ok_with_empty_content` and `test_groq_complete_with_no_fallback_raises_clearly_on_empty_content` (`tests/test_reasoning_client.py`) both fail with the fix reverted (confirmed live: `git stash` on just that file, reran, both failed with the exact expected assertions; restored, both passed again).

After redeploying that fix, the owner's real `/new` reached the writer - but the resulting screenplay cast a character who never appeared in the real story ("father" instead of "son"). Traced to the real text the bot actually received: the owner's real ~900-word story, sent as a single long message, arrived at the bot **cut off mid-word** ("...لإحضار اب" instead of "...لإحضار ابني"). Telegram's own client auto-splits an outgoing message over roughly 4096 characters into several separate messages; only the first carried the `/new` command and reached `context.args`, and the rest arrived as ordinary text messages the bot had no handler for at all - silently dropped, not even logged. The writer built an entirely plausible screenplay from a truncated, different sentence, which is why the output looked wrong without looking like an error.

Fixed: `BotConversation.handle_plain_text()` (new) accumulates every ordinary, non-command message per chat into `_ChatState.pending_text`, in arrival order; `/new` with no premise of its own uses exactly that accumulated text, completely unmodified - the owner was explicit about this ("القصة يجب ان تكون كما كتبتها" - the story must be exactly as I wrote it, no shortening, no summarizing). `src/bot/runner.py` wires a `MessageHandler(filters.TEXT & ~filters.COMMAND, ...)` to it. Covered by `tests/test_bot_conversation.py::test_plain_text_messages_are_accumulated_and_new_without_args_uses_them_verbatim` and two more for clearing on a built screenplay and on `/cancel`.

That fix worked - confirmed live, the real screenplay's title line now carried the real first words of the full story, not the truncated one. But the owner's next real check (comparing the generated dialogue against the real story line by line) found a third, different bug: the writer **invented** dialogue that outright **contradicts** the premise. The real story says the ex-wife "سكتت ولم تقل لا نعم ولا لا" (stayed silent, said neither yes nor no) when first approached - the generated screenplay instead gave her an immediate reply ("ماذا تريد الآن؟ لا فائدة من الكلام"). A second line put words in her mouth about who refused contact that the real story never has her say (the real story has the narrator's own mother denying anyone ever came, and the sister exposing that lie - different characters, different lie, than what was generated).

Root cause: `_LINES_SYSTEM_PROMPT` asked only for dialogue that "flows" and is "plausible", which gave the model room to invent freely even where the premise already supplies a character's real quoted words (the real story is full of «quoted» dialogue) or explicitly states a character did NOT speak. Fixed: the prompt now explicitly requires reusing a premise's own quoted dialogue **verbatim** wherever it exists, and forbids inventing a line a character is explicitly described as not saying or responding to when the premise says they stayed silent. Covered by `tests/test_dialogue_writer.py::test_write_dialogue_screenplay_lines_prompt_requires_verbatim_quotes_and_forbids_contradiction`, mutation-tested (reverted just the prompt change, confirmed the test fails on the exact assertion, restored, confirmed green).

This is a prompt-engineering fix, not something a mocked unit test can prove end to end - the test only proves the instruction reaches the model, not that the model obeys it. The owner's next real `/new` with a quote-heavy story is the real proof.

**That redeploy also worked** (the title line now carried the real, un-truncated start of the full story). But comparing the actual generated screenplay against the real story surfaced the project's **original, founding complaint** ("انا طلبت منك وكلاء لصنع المسلسلات و الافلام ... و لم اطلب فيديو وثائقي" - I asked for agents to make series and films, not a documentary video - the entire reason `DialogueWriterAgent`/`DialoguePipeline` were built in the first place), reappearing through a side door: "اريد ممثلين يتكلمون لا راوي يروي قصة" (I want actors talking, not a narrator telling a story). The owner's real story is written as first-person memoir prose ("كنت مطلقاً لزوجتي..."), so the cast step cast the narrating "I" as a character literally named **"الراوي"** (The Narrator), and the lines step gave that character lines like "فصرت أطلقها ولا أستمع لكلامها" (So I divorced her and stopped listening to her) - a plain narrated summary of past events, exactly the documentary-style output this whole agent exists to prevent, not a line anyone says to anyone in a scene.

Fixed in both prompts: `_CAST_SYSTEM_PROMPT` now explicitly forbids a "narrator"/"راوي" role - a first-person memoir narrator must be cast as a real character (by name if given, otherwise by their real role, e.g. "الزوج") who appears in scenes. `_LINES_SYSTEM_PROMPT` now explicitly forbids any line that summarizes or recounts past events in past tense, and instructs the model to dramatize the premise's own already-quoted scenes instead. Covered by `tests/test_dialogue_writer.py::test_write_dialogue_screenplay_prompts_forbid_a_narrator_role`, mutation-tested the same way as the fix above (reverted, confirmed the test fails on the exact assertion naming "راوي"/"narrator", restored, confirmed green). Same caveat as above: this proves the instruction reached the model, not that the model obeys it - the owner's next real result is the actual proof.

## Language and visual style, 2026-10-19

Owner asked plainly: "و ما هي اللغة التي يتكلمون بها؟" (what language do they speak?) and "اريدهم مثل المسلسلات التركية" (I want them like Turkish TV dramas).

- **Language**: `_LINES_SYSTEM_PROMPT` requires lines "in the SAME language as the premise" - since the owner's premise is Arabic, the dialogue text is Arabic. This answer was given, then immediately real-tested by the owner's own next instruction (below).
- **Visual style**: nothing previously pushed the real fal.ai generation toward any particular look - `DialogueLine.action` was sent to the model as-is, a bare stage direction ("smiling, setting the table"). `dialogue_pipeline.py` now appends a fixed `CINEMATIC_STYLE_SUFFIX` ("cinematic lighting, warm color grading, emotional close-up shot, high production value television drama") to every real prompt sent to fal.ai. This is a real code change, confirmed by test (`tests/test_dialogue_pipeline.py`, both prompt assertions now include the suffix) - **not a claim that the generated video actually looks like a Turkish drama**, which only a real generated clip can prove and none has been produced with this change yet.

## Switched the voice dialect, Tunisian → Syrian (2026-10-19)

Owner, immediately after the above: "اللهجة السورية المستخدمة في الدبلجة كانت بيضاء ومفهومة جداً لجميع سكان شمال إفريقيا" - the Syrian dialect used in [Turkish series] dubbing was "neutral/white" and very understandable across all of North Africa. Real reasoning: the whole point of this project is real reach/monetization across platforms, and Tunisian specifically is a narrower audience than the "neutral Syrian" register used industry-wide for exactly this reason (the same reason MBC and others dub Turkish series into it rather than Modern Standard Arabic or any single national dialect).

Did not guess a voice name. Checked live, same method as the original Tunisian check: `GET https://eastus.tts.speech.microsoft.com/cognitiveservices/voices/list` against this project's real Azure resource, filtered to `ar-*` locales, run from the real deployed droplet with the real `AZURE_SPEECH_KEY` already in its `.env`. Confirmed present with `Status: GA`: **`ar-SY-AmanyNeural` (female)** and **`ar-SY-LaithNeural` (male)** - alongside every other real Arabic locale Azure hosts for this resource (AE, BH, DZ, EG, IQ, JO, KW, LB, LY, MA, OM, QA, SA, SY, TN, YE - 16 countries, confirmed from the live response, not the general region table).

Changed everywhere the dialogue path's real default lives: `character_photos.py`'s `_VOICE_BY_GENDER` (the one that actually matters - every real `Character` `DialogueWriterAgent` builds gets its voice from here), `src/models.py`'s `Character.voice` fallback default, `src/config.py`'s `AZURE_TTS_LANG` default (the SSML `xml:lang` the live `DialoguePipeline` path sends - `app.py` wires `settings.azure_tts_lang` straight through), and `dialogue_pipeline.py`'s own `tts_lang` parameter default. Left untouched: `AZURE_TTS_VOICE`'s default and the OLD `EpisodePipeline`/`episode_pipeline.py` narration path's own Tunisian fixtures - neither is on the path the live bot actually runs today, and nobody asked to change the old pipeline.

`AZURE_TTS_LANG` is only a *default* read from the environment - if the deployed droplet's own `.env` already sets `AZURE_TTS_LANG=ar-TN` explicitly, that value wins regardless of this code change, so the real deploy step for this fix includes confirming (or setting) that line on the server, not just redeploying the image.

## A real, correct screenplay - and the next real gap it exposed (2026-10-19)

With all three fixes above redeployed, the owner's next real `/new 60` on the full rewritten (third-person) story produced a screenplay that, checked line by line against the real story: cast no narrator (the real cast was `الزوج`/`الأم`/`الزوجة`/`الأب`/`السائق`/`الأخت` - the actual people in the story), used the real story's own quoted wording almost verbatim line for line, correctly attributed lines to the right speaker (a real earlier mix-up - the wife's lines misattributed - did not recur), and its own last line was the real story's actual last line (the mother's death). All three fixes held up on a real run.

**Real next finding** ("انا كتبت قصة و هو يبترها" - I wrote a story and it cuts it): nothing was truncated in transit this time, but the 15 real lines a 60-second video allows were not enough room for every real quoted moment the story contains, and the model's choice of which 15 to keep was not reliably the right 15 - it skipped the single quoted line the whole plot depends on (the mother's lie that is the real reason the marriage ended: «زوجتك شتمتك وسبتك في غيابك») while keeping lines from elsewhere. This is a duration-vs-content tradeoff, not a bug the code alone can fully solve - a 1000+ word real story has more real quoted moments than 15 lines can hold. Two real responses:

1. `_LINES_SYSTEM_PROMPT` now explicitly instructs that when there are more real quoted moments than `{line_count}` allows, the ones to keep are the ones the story's actual events depend on - what starts the conflict, each real turning point, how it ends - never dropping the line that explains WHY something central happens in favor of a less essential one. Covered by `tests/test_dialogue_writer.py::test_write_dialogue_screenplay_lines_prompt_prioritizes_essential_quoted_moments`, mutation-tested the same way as the fixes above.
2. The owner's own real answer to not having enough room: raise the duration. 120 seconds ≈ 30 lines, close to or covering every real quoted moment in this particular story, at a real estimated cost roughly double 60 seconds' (`ESTIMATED_SECONDS_PER_LINE` × `PRICE_PER_SECOND_USD` in `_format_screenplay`'s own estimate line - shown to the owner before approving, same as every other duration).

Same caveat as the fixes above: the prompt change is confirmed to reach the model; whether the model reliably picks the right lines on a real run with a real quote-dense story is the owner's next real test to make, not something proven here.

## `/full` - a second mode with no selection at all (2026-10-19)

Owner, immediately after the fix above: "بل تعطيه أمر ليصنع ما كتبت له بالظبط كلمة بكلمة و لا يخرج عنها و تكون كاملة بلا نقصان و لا حذف و لا تغيير. أنا من أدفع المال و أنا حر ماذا أريد. لا يجب أن تفرض عليا أنت أو هو ما تريدان" (give it an instruction to produce exactly what I wrote, word for word, not deviating from it, complete, with no shortening, no deletion, no alteration - I'm the one paying, I'm free to choose what I want - neither you nor it get to impose what you two want). "Prioritize the essential lines" (above) is still a selection by the model - this instruction explicitly rejects that tradeoff when cost isn't the owner's concern for a given request.

Built as a genuinely different mode, not a bigger `line_count`: `DialogueWriterAgent.write_complete_dialogue_screenplay(title, premise)` never estimates or asks the model to estimate a count - it counts the real quoted dialogue moments in the premise itself with a plain regex (`«guillemets»`, `"straight quotes"`, `"curly quotes"`), then requires the model to return exactly that many lines, one per quoted moment, in order, copied verbatim, with no omission, merging, or addition - and raises `DialogueWriterError` if the model returns fewer than that real count, the same enforcement `write_dialogue_screenplay` already has for its own estimate. Reachable from the bot as `/full` (`BotConversation.handle_full`, wired in `runner.py`) - no seconds argument, same accumulated-text and approve/cancel flow as `/new`. The owner pays for however many real lines that turns out to be for their actual story - shown in the same cost estimate before approval - which is the explicit tradeoff this instruction asks for, not a decision this code makes for them.

Covered by `tests/test_dialogue_writer.py::test_write_complete_dialogue_screenplay_requires_exactly_one_line_per_real_quoted_moment` (+ 2 more) and `tests/test_bot_conversation.py::test_full_builds_a_screenplay_with_no_duration_and_one_line_per_real_quote` (+ 2 more), mutation-tested together (reverted both files, confirmed all 6 new tests fail - 3 with `AttributeError: no attribute 'handle_full'`, 3 inside `DialogueWriterAgent` - restored, confirmed green).

Same caveat as every prompt-level fix above: proven to reach the model and to enforce the right COUNT; whether the model's own copy of each quoted line is byte-for-byte verbatim on a real run is the owner's next real test, same as always.

## The first real Approve press, and two real bugs it found (2026-10-19)

The owner pressed "✅ موافقة" on a real 22-line `/full` screenplay at 19:46. Nothing came back - not a video, not an error - for over two hours. Diagnosed from the real container logs (`docker compose logs`, filtered past the constant Telegram long-polling noise) and the real fal.ai dashboard (Credits still exactly **$10.00**, unmoved - the decisive clue that this failed before spending anything, not that it was slow):

```
ERROR:asyncio:Task exception was never retrieved
...DialoguePipelineError: Character 'الأم' reference_image_path
'/app/assets/placeholder_characters/female_older.png' does not exist on disk
```

Two real, separate bugs, found together:

1. **The Docker image never contained the placeholder character images.** `Dockerfile` had `COPY src ./src` and nothing for `assets/` - every unit test exercising `character_photos.py` runs directly against the real filesystem paths on disk, so this gap was invisible to the whole test suite; only a real container, built from the real image, could have caught it. Fixed: `COPY assets ./assets` added to `Dockerfile`.
2. **A failure in the one real-money step was invisible from Telegram.** `_produce_and_notify` (`src/bot/runner.py`) had no error handling around `dialogue_pipeline.produce()`; `asyncio.create_task(...)`'s exception just died as "Task exception was never retrieved" in the container's own logs - the owner's phone got nothing at all, not even an error, which is why over two hours passed with no visible outcome in either direction. Fixed: the executor call is now wrapped in try/except, and any real exception is sent to the owner as an actual Telegram message (`❌ فشل التوليد الحقيقي: ...`) - this is the one call in the whole bot that must reach the owner directly on failure, since nothing else here retries or falls back from it.

Both real, confirmed by the real traceback above, not guessed - this project's own standing rule. Not independently proven by a redeploy yet; that is the real next step, with the explicit instruction to verify `assets/placeholder_characters/` actually exists inside the rebuilt container **before** pressing Approve again, so a second real mistake does not cost another two hours of silence.

**Both fixes above confirmed live, both worked.** `assets/placeholder_characters/` had all six real PNG files inside the rebuilt container (`docker compose exec ... ls`), and the real failure this time reached Telegram directly and instantly, exactly as designed:

```
❌ فشل التوليد الحقيقي: Client error: '422 Unprocessable Entity' for url
'https://queue.fal.run/fal-ai/wan/requests/01a1560a-bd25-76d0-8f4f-9d3c3f627f79'
```

## A real, new fal.ai failure, and a real diagnostic gap it found (2026-10-19)

This is the first real run to ever get past `RealFalVideoClient`'s submit step - every earlier real attempt (see "Real end-to-end run, 2026-10-18" above) was blocked by a `403 Exhausted balance` on submit itself, so the polling leg (`GET status_url`, `GET response_url`) had never been exercised live before this. It failed with a real `422 Unprocessable Entity` on the status check.

**Could not determine the real root cause** - and did not guess one. `status_resp.raise_for_status()` (httpx's default) only reports the status line and URL in its exception message, never the response body, which is the one place fal.ai would have said WHY. The same gap existed on the result fetch. Fixed: both now check `status_code >= 400` directly and raise `FalVideoError` carrying the real response body (`status_resp.text[:500]`), matching the pattern the submit step already used - so a repeat of this exact failure is diagnosable from the owner's own Telegram message next time, not another guess. Covered by `tests/test_fal_video_client.py::test_generate_speech_to_video_raises_with_the_real_body_on_a_failed_status_check` (+1 more for the result fetch), mutation-tested (reverted, confirmed both fail on the real `RuntimeError: status 500`/`422` from the old generic `raise_for_status()`, restored, confirmed green).

**The 422's actual cause is still unknown** and will only be known from the real message the owner's next real attempt now surfaces directly in Telegram - stated plainly rather than guessed at (a malformed `image_url`/`audio_url` data URI, a fal.ai-side validation rule not documented on the model's public page, or something else entirely are all equally unconfirmed possibilities until that real text is read).

**With that diagnostic fix deployed, the real next attempt surfaced a different, worse real problem.** The owner's real fal.ai dashboard after that attempt: `Credits: $3.26` (down from a real $10.00 - **$6.74 actually spent**), and the dashboard's own numbers for `fal-ai/wan/v2.2-14b/speech-to-video` specifically: **15 real requests, 0 errors** in the last 7 days. fal.ai's own side reports every one of those requests succeeded. The Telegram message the owner got, though:

```
❌ فشل التوليد الحقيقي: fal.ai job did not complete within 600s
(request_id=01a15617-d76a-7c33-87ed-d1c13cb81242)
```

**Real root cause, found without guessing**: `DialoguePipeline.produce()` built its `clips` list across a `for` loop with no error handling inside it. The 22-line run got through roughly 15 real, successfully-paid-for lines, then one line's real fal.ai job ran past `RealFalVideoClient`'s own 600-second wait - and that one `FalVideoError` propagated straight out of `produce()`, which discarded the entire `clips` list (along with the function call itself) before ever returning anything. Every one of those ~15 real, already-paid-for, already-finished-on-disk clips was thrown away by the code, not by fal.ai - this is the real reason the owner's money felt wasted, even though fal.ai's own records show nothing was wasted on their end.

Fixed: the per-line TTS + fal.ai + ffprobe block inside `produce()`'s loop is now wrapped in its own `try/except`. A failure there is recorded as `(line, str(exc))` in a new `failed_lines` list on `ProducedDialogueEpisode` (`src/models.py`) and the loop continues to the next line, instead of aborting everything. The structural validation checks above it (unknown speaker, missing/absent reference photo) still raise immediately and stop the whole episode - deliberately: those happen before any money is spent for that line and indicate a broken screenplay, where continuing would just repeat the identical failure on every remaining line by the same character.

`BotConversation.format_produced()` now reports any `failed_lines` by their 1-indexed line number in the message the owner actually reads, so they know exactly which real money bought nothing and can `/edit` + retry just those lines rather than redoing (and re-paying for) the whole episode. `MockFalVideoClient` gained `fail_at_calls` so this is testable without a real fal.ai account: `tests/test_dialogue_pipeline.py::test_produce_keeps_already_paid_for_clips_when_a_later_line_fails` and `tests/test_bot_conversation.py::test_format_produced_reports_failed_lines_by_number`, both mutation-tested (reverted all three changed files, confirmed both fail - one on a propagated `FalVideoError`, one on `TypeError: unexpected keyword argument 'failed_lines'` - restored, confirmed green).

Not yet proven on a real run with this fix deployed - same standard as every fix above.

## A real risk found by review, not by a failure (2026-10-19)

Owner's instruction: review everything before the next real attempt, not just react to the next failure. Re-read `src/bot/runner.py` end to end looking for anything else that could fail the same silent way the two bugs above already did.

**Found**: Python's own `asyncio` documentation states plainly that the event loop holds only a *weak* reference to a `Task` created by `asyncio.create_task()` - "a task that isn't referenced elsewhere may get garbage collected at any time, even before it's done." `_produce_and_notify` (the one call that runs for minutes and spends real money) was being launched with `asyncio.create_task(...)` and the return value discarded immediately - exactly the pattern Python's docs warn against. This had not failed yet in anything observed tonight, but it is a real, documented risk for precisely the kind of long-running fire-and-forget background task this file already has two other silent-failure incidents on, from different causes.

Fixed: `build_application` now keeps a `_background_tasks: set[asyncio.Task]` and a small `_spawn()` helper that adds every task to it and removes it via `task.add_done_callback(_background_tasks.discard)` once it finishes - the standard pattern from Python's own asyncio docs for exactly this situation. `runner.py` is this project's own documented "thin, not independently unit-tested" layer (see its own module docstring, and nexus-commerce's matching convention) - this fix is a direct application of CPython's documented guidance, not a guess, but like the rest of this file's real-integration code, confirmed only by passing `python3 -c "import ast; ast.parse(...)"` plus the full suite staying green, not by a dedicated unit test.

## A dedicated code review before the next real attempt (2026-10-19)

Owner's instruction, repeated and explicit ("لن اكف عنك الا ان تراجع كل شيء و تكتشف الاخطاء" - I won't stop until you review everything and find the bugs): rather than more ad-hoc reading, ran this project's own `code-review` skill at high effort against every commit from the DialoguePipeline bot rewrite onward (`d349fca~1...704c64d`). It found four real, previously undetected defects - none guessed, each with a concrete failure scenario:

1. **`_new`/`_full`/`_edit` had no error handling at all.** If `DialogueWriterAgent`'s reasoning call failed (Groq and its Gemini fallback both failing, or an unrecoverable malformed reply), the exception propagated out of the handler with nothing sent to Telegram - the exact "owner gets total silence" failure class the Approve-button fix (above) exists to prevent, just reachable from a different command it never covered. Fixed: `application.add_error_handler(_on_error)` registered in `runner.py` - python-telegram-bot's own documented mechanism for exactly this, catching an uncaught exception from *any* handler and reporting it to the chat it happened in, rather than wrapping every handler individually.
2. **`write_complete_dialogue_screenplay`'s `max(1, _count_quoted_moments(premise))` floored a genuinely quote-free premise at 1** - telling the model "the premise contains exactly 1 directly quoted moment" when it contains zero, forcing it to invent the one thing `/full` exists to guarantee never happens. Fixed: raises a clear `DialogueWriterError` instead, telling the owner to use `/new` for a premise with no ready-made quotes.
3. **`_parse_cast` only checked that `name`/`gender`/`age_group` keys existed, never that their values were real strings.** A reply like `{"gender": null}` passed validation here and crashed deeper inside `character_photos.py`'s `None.strip()` with an unclear `AttributeError` instead of this function's own clear message. Fixed: now requires each value to be a non-empty string.
4. **`DialoguePipeline`'s per-line failure handling (added two fixes ago) let `_format_failed_lines` claim every failed line's cost "was not counted" - false whenever the real, billed `generate_speech_to_video()` call had already succeeded and only a later local step (writing the file, `ffprobe`) failed afterward.** Fixed: a `billed` flag set the instant the real paid call returns successfully; a failure after that point carries an explicit "real money was spent at fal.ai before this failure - check the fal.ai dashboard" note instead of the blanket false claim, and `conversation.py` now shows each failed line's own real message instead of one summary claim.

All four covered by new tests (`tests/test_dialogue_writer.py::test_write_complete_dialogue_screenplay_raises_on_a_premise_with_no_real_quotes`, `::test_parse_cast_rejects_a_null_field_value`, `::test_parse_cast_rejects_an_empty_string_field_value`, `tests/test_dialogue_pipeline.py::test_produce_reports_when_a_line_was_billed_by_fal_ai_before_the_failure`), mutation-tested together (reverted all three changed source files, confirmed all four new tests fail with four different real assertion errors, restored, confirmed the full 119-test suite green). The error-handler fix (#1) is `runner.py`'s own documented not-independently-unit-tested layer, same as the asyncio-task fix above - confirmed by syntax check and the suite staying green, not a dedicated test.

**Still not independently proven by an actual real run** - same standard as every fix in this file. Every fix above removes a specific, named risk found either by a real incident or by this review; none of them is a claim that the system is now bug-free.

## Automatic deploy (CI/CD), 2026-10-20

Owner's instruction ("اريد ان استغنى عنك... في التصليح و الاصلاح" - I want to stop needing you for every fix/redeploy): every single fix tonight needed the owner to manually copy a command, open the DigitalOcean browser console, and paste it. `.github/workflows/deploy.yml` ends exactly that one step - on every push to `main`, GitHub itself (not this sandbox, which has no outbound SSH - see "Real deployment" above) connects to the real droplet over SSH and runs the same `git pull && docker compose up -d --build nexus-studio` the owner was pasting by hand all night, then prints the post-deploy logs into the GitHub Actions run itself so a failure is visible there directly.

Uses `appleboy/ssh-action@v1.2.4` - checked live, not guessed, against a real December 2025 dependency-bump record (a `web-php` project's own CI pinned it at that exact version); an earlier draft would have pinned an invented `v1.0.3`.

**This does not remove the need for a real fix to be written and tested first** - stated plainly, since the owner's own next question was literally whether this replaces needing that at all. It only removes the manual deployment step after a fix already exists, is tested, and is pushed - the exact gap between "I pushed a fix" and "the owner's server is actually running it" that cost real back-and-forth tonight.

**Setup still needed, not done yet** (requires the owner's real droplet and real GitHub repo settings, neither reachable from here without them):
1. The real SSH *public* key (`github-actions-deploy-nexus-studio`, generated this session, ed25519) must be added to the droplet's `~/.ssh/authorized_keys` for the `root` user, via the real DigitalOcean console.
2. Three real GitHub Actions repository secrets must be set (`Settings → Secrets and variables → Actions` on `hichembs2019-dev/nexus-studio`): `DEPLOY_HOST` (`159.89.22.193`), `DEPLOY_USER` (`root`), `DEPLOY_SSH_KEY` (the matching private key, generated this session, never committed to the repository).
3. **Tested live, three times, still failing as of this writing.** Triggered `workflow_dispatch` runs directly via the GitHub API and read the real job logs each time (`get_job_logs`), rather than asking the owner to report back: all three runs failed identically with `error: missing server host` and every `INPUT_HOST`/`INPUT_USERNAME`/`INPUT_KEY` env var logged completely empty, even after the owner confirmed editing the existing `DEPLOY_HOST`/`DEPLOY_USER`/`DEPLOY_SSH_KEY` secrets (not creating new ones - a real `"a secret with the same name already exists"` error on the first attempt confirmed the names were already registered). The real cause is still unconfirmed - plausibe candidates, not yet distinguished: the secrets were added to a different repository, to Environment secrets rather than Repository secrets (this workflow declares no `environment:`, so it cannot see environment-scoped secrets), or to Codespaces/organization secrets instead of the repo's own Actions secrets. The real next step is confirming the exact URL/repo name the owner is editing on, not guessing further.

## `/edit` accepts several edits in one message (2026-10-20)

Owner's instruction ("لماذا كتبت يعدل مشهد واحد؟ بل يعدل كل المشاهد فالحرية لي و ليست لكم" - why did you write it edits one scene? it should edit ALL scenes, the freedom is mine, not yours): `/edit` previously accepted exactly one `<number> <text>` pair per command - editing a multi-line screenplay meant one `/edit` message per line.

Real mechanical snag found while building this, not guessed: Telegram's `context.args` (what every other command here reads) splits the message text on *any* whitespace, including newlines - so `/edit\n1 a\n3 b` would arrive as five separate tokens with no way to tell where one edit's text ends and the next line number begins. `runner.py`'s `_edit` handler now reads `update.message.text` directly and splits it only once, on the first whitespace run (the one right after `/edit` itself) - `str.split(None, 1)` - which keeps every real newline in the remainder intact. `BotConversation.handle_edit()` then splits that remainder on real newlines, parses each non-blank one as `<number> <text>`, applies every valid edit in a single pass, and reports back the one updated screenplay - with any line numbers that didn't exist in the draft listed by number rather than silently ignored.

Covered by `tests/test_bot_conversation.py::test_edit_updates_several_lines_from_one_multiline_message` (+2 more for partial not-found reporting and blank-line tolerance), mutation-tested (reverted `conversation.py`, confirmed all three new/changed tests fail with three different real assertion errors, restored, confirmed the full 122-test suite green).

### What is NOT built yet (stated plainly, not implied)

- **Multi-platform publishing** (TikTok/Instagram/X/Snapchat/Twitch/Kick)
  is still not built - unchanged from the earlier entry above, YouTube
  is still the only real publishing client.
- **The bot IS now deployed** (see "Real deployment, 2026-10-19" above) -
  this entry was true earlier the same day (no process was running it
  anywhere, which is why an earlier real story sent to the bot got no
  reply at all) but is now out of date; left here with a correction
  rather than silently deleted, per this project's own standing rule
  about not erasing a claim that turned out to be wrong.
- **No real `DialoguePipeline.produce()` call has been made from the
  bot.** The owner's standing instruction ("لا تفعل شيء بالمال و لا
  تخسره" - don't do anything with the money, don't lose it) means the
  Approve button's real fal.ai spend should only be pressed once the
  owner is ready for it to happen, with the screenplay's estimated cost
  already shown to them beforehand (`_format_screenplay`'s estimate
  line) - this is a real-money action gated by human approval by design,
  not yet exercised live.

## Automatic deploy (CI/CD) - first real success, and a real secret leak found in its own log (2026-10-20)

**Resolution of the "tested live, three times, still failing" entry above.** After the owner re-entered all three secrets (`DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`) via the pencil/edit icon on the existing entries - the earlier failures were a genuinely empty/malformed secret value each time, confirmed from the real job logs each time, never guessed - a `workflow_dispatch` run (`run_number: 7`, `id: 39400343284`) finished with `conclusion: "success"`. Read directly from the run's own job log (`get_job_logs`), not inferred from the green checkmark alone:

- SSH connected (`appleboy/ssh-action@v1.2.4`, step duration 16.7s - the first run that took longer than a few seconds, because it was the first one that got past authentication).
- `git pull origin main` fast-forwarded `2980282..27b37c3`, the real latest commit (multi-line `/edit`), 9 files changed - printed in the log, not assumed.
- `docker compose up -d --build nexus-studio` rebuilt the image, including `COPY assets ./assets` (step 7/7, the exact step whose absence caused the very first real Approve failure - see "Real deployment, 2026-10-19" above) - `Container nexus-studio-nexus-studio-1 Recreated` → `Started`.
- The bot's own post-deploy log, captured by the workflow's own `docker compose logs --tail=30`, shows a real successful boot: `nexus-studio bot starting (long-polling)`, `POST .../getMe "200 OK"`, `POST .../deleteWebhook "200 OK"`, `Application started`.

This is the first push-to-deploy cycle that needed zero manual console steps end to end.

**Real secret leak found in reading that same log.** The `getMe`/`deleteWebhook` log lines above printed the full Telegram API URL - `https://api.telegram.org/bot<TOKEN>/getMe` - with the live bot token in plain text, inside a GitHub Actions run log. Root cause: `src/main.py`'s `logging.basicConfig(level=logging.INFO)` left `httpx`'s own logger at INFO, and httpx logs the complete request URL at that level; the container prints that to stdout, and the workflow's own `docker compose logs` step (added so a deploy failure is visible without SSHing in by hand) relays stdout straight into the run log. GitHub only masks values it knows are `secrets.*` in the *workflow's own* `with:`/`env:` block (`DEPLOY_HOST`/`DEPLOY_USER`/`DEPLOY_SSH_KEY` show as `***` in the same log) - it has no way to know a string appearing inside a shell command's own output is a secret too.

Fixed at the root, not by hiding the symptom: `logging.getLogger("httpx").setLevel(logging.WARNING)` added right after `basicConfig` in `src/main.py`, so httpx never logs the token-bearing URL at any level anywhere again - in this log, a future one, or a local run. Covered by `tests/test_main_logging.py::test_httpx_logger_is_not_info_level`, which imports `src.main` and asserts the `httpx` logger's own `.level` is at or above `WARNING`; mutation-tested (reverted `src/main.py`, confirmed the test fails with `httpx logger is at NOTSET`, restored, confirmed the full 123-test suite green).

**The already-leaked token itself is not fixed by this change** - the fix stops the next leak, not the one that already happened. The owner needs to rotate the real bot token via @BotFather (`/revoke` or `/token` on the existing bot) and update `TELEGRAM_BOT_TOKEN` on the real droplet - this requires the owner's own Telegram account and cannot be done from here.

## Real concurrent-deploy collision, found immediately after the first real CI/CD success (2026-10-20)

Minutes after the first real push-triggered deploy succeeded (run #8, `conclusion: success` - the httpx-logging fix above, confirmed live via its own job log: `git pull` fast-forwarded, image rebuilt, container recreated and started), a second run (#9, a manual `workflow_dispatch` fired to double-check the result) landed on the same droplet while #8 was still mid-`docker compose up -d --build`. Read from #9's own real job log, not guessed:

```
Container nexus-studio-nexus-studio-1  Recreate
Error response from daemon: Conflict. The container name
"/b35aba93a2c4_nexus-studio-nexus-studio-1" is already in use by
container "0873e46dec97...". You have to remove (or rename) that
container to be able to reuse that name.
Process exited with status 1
```

Not a code defect - a real gap in `deploy.yml` itself: nothing stopped two deploys from running at once, and Docker's own rename-during-recreate step collided when they did. Fixed with a `concurrency: {group: deploy-droplet, cancel-in-progress: false}` block - a second run now queues behind the first instead of racing it (queuing rather than cancelling, specifically, because a cancelled mid-deploy run could land between `docker compose up -d --build` and the `docker compose logs` read that is this workflow's whole reason for existing, silently losing the evidence a real failure needs).

## Token leak - closed (2026-10-20)

Owner rotated by creating a new bot (`t.me/Studio_Nexus_bot`, new token) via @BotFather and **deleting the old leaked bot entirely** (confirmed by the owner: "صار delete للقديم") - stronger than a plain revoke, since a deleted bot cannot be un-deleted back into a working state by anyone who still has the old token.

New token applied on the real droplet via a one-line `sed` replace of `TELEGRAM_BOT_TOKEN` in `.env` (no manual `nano` edit needed), followed by `docker compose up -d --build`. Real console output, photographed by the owner:

```
nexus-studio-1  | INFO:nexus_studio.main:nexus-studio bot starting (long-polling)
nexus-studio-1  | INFO:telegram.ext.Application:Application started
```

No `getMe`/`deleteWebhook` URL line appears - confirming the `httpx` logging fix above holds on the real deployed container, not only in the test suite. `Application started` with no earlier `Unauthorized`/auth error also confirms the new token itself is valid - python-telegram-bot's `run_polling()` calls `getMe` during startup and would have raised before this line if the token were wrong.

Full timeline of this incident, real evidence at every step: leak found reading a real CI log → root-caused to `httpx` INFO logging → fixed and mutation-tested → fix verified live on first deploy (no token line) → owner deleted the leaked bot and created a replacement → new token deployed and verified live, bot running.

## fal.ai job timeout - real incident, root cause, and fix (2026-10-20)

Real failure reported by the owner with evidence from three sources at once: two Telegram bot messages showing the exact same failure twice in a row -

```
تم توليد 0 مقطع لكن تعذّر دمجها. التكلفة الحقيقية حتى الآن: $0.000
⚠️ فشلت 1 أسطر - يمكنك تعديلها وإعادة المحاولة:
- سطر 1: fal.ai job did not complete within 600s (request_id=01a158d8-...)
```

- a fal.ai dashboard screenshot (`Credits: $2.78`, `Errors in the last 7 days: 0`, `Requests in the last 7 days: 15`) showing fal.ai's own side recorded **no error** for either attempt, and
- a PayPal screenshot the owner read as "I paid and got nothing" - which on inspection is unrelated infrastructure billing (DigitalOcean droplet hosting, GitHub, Shopify), not fal.ai at all; the real fal.ai accounting for this specific incident is the bot's own `$0.000` line above (`DialoguePipeline` only prefixes a failed line with a "real money was spent" note when the `generate_speech_to_video` call itself succeeded - see "Clips discarded on partial pipeline failure" above - and it did not here).

**Root cause, not guessed - read from the client code and fal.ai's own docs:**
1. `RealFalVideoClient.generate_speech_to_video()` gave up polling at a flat 600s. The closest real timing figure found (a different host's own deployment of the same underlying Wan 2.2 speech-to-video model, since fal.ai publishes no published median for its own endpoint) reports a **median** successful run of ~412s - meaning 600s sat barely above the midpoint of the real distribution, so a routine (not rare) slower-than-median run would hit the cap and fail outright even with fal.ai working completely normally - consistent with fal.ai's own dashboard showing 0 errors on their side for both failures.
2. **Separately, a real risk found while fixing this**: giving up at the deadline only stopped *this client* polling - fal's own queue docs state a queue-based job keeps running after the connection drops, and "only an explicit cancel API call can stop the request." The client captured a real `cancel_url` at submit time and never used it, so an abandoned job could keep running on fal's side and be billed with no video ever collected here - the worst version of exactly what the owner feared ("ادفع بدون نتيجة").

**Fixed, both together:**
- `POLL_TIMEOUT_S` raised from 600 to 1200 (real margin above the real ~412s median figure, not an arbitrary round number).
- On timeout, `PUT <cancel_url>` is now sent (real HTTP method confirmed from fal's own queue docs: `PUT https://queue.fal.run/.../requests/{id}/cancel`) before raising - this removes a still-queued job outright (fal's docs: 200, never runs) and best-effort signals an in-progress one to stop, wrapped so a failure to cancel never hides the real timeout error.

Covered by `tests/test_fal_video_client.py::test_generate_speech_to_video_cancels_the_real_job_on_a_real_timeout`, which fakes `time.monotonic()` to reach the deadline after exactly one poll (no real 1200s wait in the test) and asserts the real `cancel_url` was PUT. Mutation-tested: reverted the fix, confirmed the test fails with the old `600s` message, restored, confirmed the full 124-test suite green.

**What this does NOT fix**: if fal.ai's own queue is genuinely overloaded well past 1200s, a job can still time out - this raises the margin to match real-world timing, it does not guarantee completion. The owner should retry the specific failed line with `/edit` rather than re-running the whole screenplay, which is exactly what the per-line `failed_lines` reporting (see above) exists for.

## Merge-failure clip loss and a hard spend ceiling (2026-10-20)

**Real incident the owner reported**: fal.ai's own Usage page showed $8.18 billed for 40.91s of real completed `fal-ai/wan/v2.2-14b/speech-to-video` output, but the owner never received any video from the bot ("قلت لك لم استلم شيء").

**Root cause, read from the code, not guessed**: `DialoguePipeline.produce()`'s final merge step -

```python
stitched_path = self._stitch(clips, slug) if len(clips) > 1 else (clips[0].local_path if clips else None)
```

- sat *outside* the per-line `try/except` that already protects real, already-paid clips from a later line's failure (see "Clips discarded on partial pipeline failure" above). When `_stitch()` (an `ffmpeg concat` call) raised for any reason, that exception escaped `produce()` entirely, was caught generically by `runner.py`'s `_produce_and_notify`, and reached the owner as "❌ فشل التوليد الحقيقي ... لم يكتمل أي مقطع" - **false** whenever clips had genuinely finished and been billed, and worse than useless: it actively hid that real money had bought something real which was simply never handed over.

**Fixed**: the merge call is now wrapped in its own `try/except` inside `produce()`. On a merge failure, `ProducedDialogueEpisode` is still returned with every real clip intact, `stitched_path=None`, and the real failure reason in the new `stitch_error` field. `BotReply` gained `video_paths: list[str]` (plural) alongside the existing single `video_path`, and `format_produced()` now sends every real, already-paid clip individually when merging fails, instead of describing them in text with no video attached at all. `runner.py`'s `_send_reply` sends each one.

Covered by `tests/test_dialogue_pipeline.py::test_produce_keeps_real_paid_clips_when_the_final_merge_fails` (fakes a real `ffmpeg concat` failure via `subprocess.run`, asserts both real clips and the real error message survive) and `tests/test_bot_conversation.py::test_format_produced_without_a_stitched_path_still_delivers_the_real_paid_clips` (asserts `reply.video_paths` actually carries the real clip paths). Mutation-tested: reverted both files, confirmed all new/changed tests fail with real assertion/attribute errors, restored, confirmed the full 128-test suite green.

**Separately, the owner's own standing instruction, repeated explicitly** ("قلت لك ضع له كود يشتغل و يسحب المال الا عند صنع فيديو" - put code in that only draws money when a video is actually made): fal.ai itself already only bills for real completed output - confirmed live twice now (the two $0.000 timeout failures above, and this incident's own $8.18 matching exactly 40.91 real completed seconds, not a single cent for either earlier timeout). That was never the actual gap. The actual gap was a **second, independent** request: a hard ceiling on total real spend *within one production run*, enforced in code, separate from the cost estimate already shown before Approve.

Added: `Settings.max_episode_cost_usd` (env `MAX_EPISODE_COST_USD`, default `10.0`), passed into `DialoguePipeline(max_cost_usd=...)`. Before attempting each line, `produce()` now checks total real spend so far against the ceiling; once reached, every remaining line is skipped **without ever calling fal.ai** - zero further real money spent - and lands in the new `skipped_lines` field (distinct from `failed_lines`, which is for a real attempt that failed, not one never attempted). `format_produced()` reports skipped line numbers explicitly so a shorter-than-expected result is never unexplained.

The ceiling is checked *before*, not after, a line - a line's real cost is only known once fal.ai has actually run it, so the guard cannot predict and block the exact line that would cross the ceiling; it guarantees no *further* line is attempted once already over it, capping runaway spend to at most one line's cost past the configured limit rather than leaving it unbounded.

Covered by `tests/test_dialogue_pipeline.py::test_produce_stops_spending_once_the_hard_cost_ceiling_is_reached` (asserts the real fal.ai client is called exactly once, not three times, against a 3-line screenplay with a ceiling below one line's cost) and `tests/test_bot_conversation.py::test_format_produced_reports_lines_skipped_by_the_hard_cost_ceiling`. Mutation-tested the same way; full suite green at 128/128.

## Delivery after production was itself unprotected (2026-10-20)

Found by code review following the owner's explicit demand to close every remaining way real money could be spent for nothing ("افعل كل شيء لكي لا نخسر المال ... اي شيء يخسر المال بدون فائدة لا"): `_produce_and_notify` in `runner.py` only wrapped the `dialogue_pipeline.produce()` call itself in a `try/except` - building the reply (`format_produced`) and actually sending the video(s) to the owner (`_send_reply`, which calls Telegram's `send_video`) ran completely unprotected, AFTER real money had already been spent and real clips already existed.

If `send_video` ever raised for any reason - Telegram's own file-size limit (~50MB on the Bot API), a network error, anything - the background task would simply die with no message sent at all: the exact "owner gets total silence" failure class the 2026-10-19 fix (`_produce_and_notify`'s own try/except around `produce()`) exists to prevent, reachable again from a different point, and specifically worse here because it would happen *after* real spend rather than before it.

Fixed: `format_produced()` + `_send_reply()` are now inside their own `try/except`. On failure, the owner is told plainly that production succeeded (with the real cost and clip count) but delivery failed, with the real exception - never silence.

Not covered by an automated test - `runner.py`'s own docstring states this file is deliberately kept untestable-without-a-live-bot (same split `nexus-commerce`'s `telegram_listener.py` uses), consistent with how every other fix in this file has been verified (syntax check + the full test suite staying green, since nothing in it imports `runner.py`). Verified by `python3 -c "import ast; ast.parse(...)"` and the full 128-test suite staying green (unaffected, as expected - no test exercises this file).

## Real, story-specific character photos replace the fixed six-photo panel (2026-10-20)

Real feedback after the first video actually reached the owner (see "Automatic deploy... first real success" and the fal.ai timeout section above - the delivery pipeline itself was being validated with a tiny, deliberately cheap real test at the owner's own $1.82 remaining balance): the character's face in that real video was one of the six hand-drawn cartoon placeholders (`assets/placeholder_characters/`), assigned by `resolve_placeholder_photo(gender, age_group)` - a deliberate choice from earlier the same day ("الصور طبعا افتراضية... تُسند تلقائيًا حسب وصف كل شخصية"), made before anyone had actually seen it in a real produced video. Once seen, it was rejected outright, with a real reference photo attached showing the photorealistic style wanted instead ("هذه هي الصور التي اقصدها ... لا صور متحركة" - these are the photos I mean, not cartoon ones).

**First design offered (quoted to the owner with its real cost) was wrong and the owner corrected it immediately**: generate the same six generic (gender × age_group) photos once with AI and cache them forever, reused across every future story exactly like the cartoons were. Owner's correction: "لا تكون صور ثابتة الى الابد ... بل حسب القصة و السيناريو" (not fixed forever - according to the story and screenplay) and "الشخصيات تُولد حسب السيناريو" (characters are generated according to the screenplay) - two different stories' "a middle-aged man" characters must not share a face.

**Built instead**: the cast-writing LLM call (`dialogue_writer.py`'s `_CAST_SYSTEM_PROMPT`) now also infers a short `appearance` description per character (hair, build, distinguishing features) from THAT premise - validated the same non-empty-string way `name`/`gender`/`age_group` already are. `DialoguePipeline.produce()` generates one real photorealistic portrait per speaking character from their own `appearance`, in a pre-pass before any line is produced, via a new `fal-ai/flux/dev` client (`image_client.py`, contract verified live via WebFetch of fal's own model page 2026-10-20: endpoint, `$0.025/megapixel` pricing, `images[0].url` response shape - not guessed). The cheap cartoon placeholder stays as the free preview shown before Approve and as the fallback if real generation fails for a character - production still continues for everyone else rather than aborting.

Cost discipline carried over from every fix above:
- Cached per **(this screenplay, this character)** - `{output_dir}/characters/{episode_slug}_{character_slug}.jpg` - so a retry of the same screenplay (e.g. after `/edit`) never re-bills an already-generated portrait, while two unrelated stories' characters still get distinct photos.
- Counted toward `Settings.max_episode_cost_usd` the same as line costs - the ceiling is checked before each character's portrait too, so a cast of many characters cannot silently blow past the configured ceiling before a single video line is even attempted.
- A real generation failure for one character never aborts the episode - it falls back to that character's placeholder and is reported to the owner by name (`portrait_failures`), distinct from a line that failed.
- Reuses the fal.ai timeout-cancel fix from earlier the same day (`image_client.py`'s `RealFalImageClient` cancels an abandoned job on timeout exactly like `fal_video_client.py` does) rather than duplicating the earlier mistake in a second client.

Covered by `tests/test_image_client.py` (submit/poll/download, failure, and the same real timeout-cancel contract as the video client), `tests/test_dialogue_writer.py` (appearance validated and carried onto `Character`), and `tests/test_dialogue_pipeline.py` (real portrait generated and actually used in the fal.ai video call's `image_data_uri`, cached across a second `produce()` call without regenerating, falls back to the placeholder on a real failure without aborting, respects the cost ceiling). Mutation-tested: reverted each file in turn, confirmed every new/changed test fails (an import error for the pipeline file - the new symbol does not exist at all without the fix - and real assertion failures for the writer and conversation files), restored, confirmed the full 141-test suite green.

## Outgoing replies over Telegram's real message limit were never chunked (2026-10-20)

Owner's question after reviewing a real long screenplay (8 scenes, originally 26 real quoted lines): "لماذا النهاية متقطعة في البوت؟ لماذا لا يكون كله في ميساج واحد؟" (why is the ending cut off in the bot, why isn't it all in one message?).

**Real mechanism confirmed, not guessed** (checked against real developer reports of Telegram's Bot API, since the official docs page wasn't directly searchable): `sendMessage` has a real 4096-character limit and, unlike the Telegram *app* auto-splitting what a human types into the compose box (already handled here - see the `/new`-truncation fix earlier, `handle_plain_text`'s accumulation), the Bot API itself does **not** split an oversized outgoing call for you - it rejects the whole call with `Bad Request: message is too long`. `runner.py`'s `_send_reply` sent `reply.text` - the full formatted screenplay or produced-result message, which grows with cast size, line count, and any failed/skipped/portrait-failure notes - in one single, unchunked call.

**Also found while fixing this**: the earlier cast cap of "2 to 6 characters" in `dialogue_writer.py`'s `_CAST_SYSTEM_PROMPT` was too low for a real story the owner wrote (8 genuinely distinct speaking roles - protagonist, his own mother and father, his ex-wife, her father, his sister, a taxi driver, a neighbor). Forcing a 6-character cap on an 8-speaker story would have made the model merge two characters into one, misattributing a real quoted line to the wrong speaker - exactly the correctness `/full` exists to guarantee. Raised to 10.

**Fixed**: `split_for_telegram()` (`conversation.py`) chunks any outgoing text at the last real newline before a conservative 4000-character limit (kept below the real 4096 cap for margin, since Telegram is reported - not officially documented - to count each newline as 2 of the 4096), falling back to a hard split only when a single run has no newline to break on at all. `runner.py`'s `_send_reply` now sends every chunk in order, with the reply's buttons attached only to the last one. Covered by `tests/test_bot_conversation.py` (under-limit text stays one chunk, a long real-shaped text splits with no chunk over the limit and rejoins back to the exact original, a newline-free run hard-splits cleanly, the real default limit stays under 4096). Mutation-tested: reverting `conversation.py` makes the whole test module fail to import (the function doesn't exist at all without the fix) - restored, confirmed the full 145-test suite green.

## Silent quote-count corruption - a real story's lines went missing with no error (2026-10-20)

Real incident, found from the owner's own screenshot: `/full` on a real 8-scene, ~23-real-quote story built a screenplay with only 15 lines ($6.00 estimate) - fewer than half the real quoted moments in the text - with **no error at all**. A second `/full` sent right after (without resending the story) then failed with "لم أجد أي اقتباس حقيقي" - correct, documented behavior (`handle_full` clears `pending_text` after a successful build - the story has to be resent for a second attempt), not a bug, but confusing on top of the silent undercount.

**Root cause, reproduced, not guessed**: `_QUOTED_MOMENT_PATTERN` (`"[^"]+"` etc.) matches from one quote mark to the next one it finds. If a single quote character anywhere in a long pasted premise is dropped, duplicated, or silently converted by a phone's keyboard/clipboard into a different character while pasting (straight `"` vs curly `“”` are genuinely different characters a mobile autocorrect can swap), the regex does not fail - it just keeps matching from the next stray opening quote all the way to whatever closing quote it finds *next*, silently swallowing every real line in between with zero warning. Reproduced directly:

```python
>>> pattern.findall('"line one" "line two" رحمها الله" "line four"')
['"line one"', '"line two"', '" "']
# "رحمها الله" text in the gap is gone entirely, with no error
```

This is the exact opposite of the owner's own most-repeated standing rule ("يجب ان ينفذ كل حرف و كل كلمة كتبتها بدون حذف او نقصان" - every letter and word must be used, nothing dropped) - and worse than a loud failure, because nothing in the bot's own reply reveals it happened; the owner only found it by counting.

**Fixed**: `_count_quoted_moments()` now validates the premise's own quote marks BEFORE counting - an odd number of straight `"` quotes, or a mismatched open/close count of curly `“”` or guillemet `«»` quotes, each PROVE at least one quote is unpaired somewhere, and now raise a clear, real error asking the owner to check their text and resend, instead of silently losing whatever fell inside the broken span.

Covered by `tests/test_dialogue_writer.py` (odd straight-quote count, mismatched curly quotes, mismatched guillemets - each asserts the real error and that the model was never even called, matching the "never risk building from a corrupted count" principle the zero-quotes check above already established). Mutation-tested: reverted `dialogue_writer.py`, confirmed all three new tests fail (the mock reasoning client runs out of scripted replies because the model gets called when it shouldn't), restored, confirmed the full 148-test suite green.

**What this does NOT fix**: it cannot recover the lost text or tell the owner which specific quote mark is the problem - only that the count proves one is broken somewhere. The owner still has to visually check the pasted text for a missing or stray quotation mark.

## On-screen text captions - a real, zero-cost way to cut a line's cost (2026-10-20)

Owner's instruction while reducing a real, expensive screenplay's cost ("نحذف الام و يبقى حوارها مكتوب على الشاشة... و كذلك نحذف الاب... في صورة الابن و الاب و الام... لكي نستحقظ على المال" - remove the mother as a speaking role, her dialogue stays written on screen; same for the father; in an image of the son, father and mother together; to save money): a genuinely new, real capability, not a text edit to the premise.

**Verified live before writing any pipeline code, not assumed**: ffmpeg's `drawtext` filter needs an actual font *file* with Arabic glyph coverage plus libfribidi (for correct right-to-left reordering) to render Arabic text at all. This project's `Dockerfile` (`apt-get install ffmpeg`) pulls in **no fonts whatsoever** - confirmed by listing installed fonts before adding anything. `fonts-noto-core` (a real, confirmed-installable Debian/Ubuntu package) ships `NotoNaskhArabic-Regular.ttf`; a real test render with it was extracted to a frame and read - properly shaped, right-to-left, readable Arabic - before this was built into the pipeline. Added to `Dockerfile`.

**Built**: `DialogueLine.spoken: bool = True` - a line with `spoken=False` costs **nothing real at all**: `DialoguePipeline.produce()` skips TTS and the real fal.ai speech-to-video call entirely for it, and instead renders it locally as on-screen text (`_render_caption_clip`, ffmpeg `drawtext`) over a shared group photo (`_build_caption_group_image`, a free local `ffmpeg hstack` of the already-resolved real portraits of every character with at least one caption line - built once per screenplay, reused for every caption in it). Caption duration is estimated from the line's own character count (`CAPTION_CHARS_PER_SECOND`, floored at `CAPTION_MIN_SECONDS`), since there is no real TTS audio to measure it from.

New bot command `/caption <line number> [more numbers]` (`BotConversation.handle_caption`, same multi-number-per-message pattern `/edit` already uses) flips specific lines to `spoken=False` on the current draft, **before** Approve - `_format_screenplay` marks each one `(نص على الشاشة - بدون صوت، تكلفة $0)` and excludes it from the pre-approval cost estimate, so the owner sees the real savings before committing to anything.

**Known limitation, stated plainly**: the group photo includes every character with at least one *caption* line, not every character merely present in that scene - in the real story that prompted this, the son's own portrait is not automatically added to the mother/father group photo just because he is also in the scene, since nothing in the current data model records "who else is on screen for a spoken line." If a speaking character needs to visually appear in a caption's group photo too, one of their own lines would need marking `/caption` as well, or this needs a real per-scene shot model - a bigger change intentionally left out of this pass.

Covered by `tests/test_dialogue_pipeline.py` (a `spoken=False` line costs `$0.00`, calls neither TTS nor fal.ai, produces a real playable clip at least `CAPTION_MIN_SECONDS` long - skipped automatically on a machine without `fonts-noto-core` installed, so the suite stays portable outside the real Docker image; a real local caption-render failure is reported without the "real money was spent" prefix the billed path uses, since none was) and `tests/test_bot_conversation.py` (`/caption` marks a line and reflects the real lower estimate, out-of-range numbers reported, no-draft and no-numbers usage messages). Mutation-tested: reverting `dialogue_pipeline.py`/`models.py` together breaks the whole test module's import (the new symbols don't exist at all), reverting `conversation.py` makes all four new `/caption` tests fail with `AttributeError` - restored both ways, confirmed the full 154-test suite green.

## `/new` versus `/full` confusion caused a real $6.838 wrong video, published live (2026-10-20/21)

Real severe incident, confirmed only once the owner forwarded a screenshot of their own actually-sent Telegram message: a real "15 lines, $6.00" screenplay had recurred three separate times, each one wrongly suspected to be the quote-count-corruption bug fixed above (and that fix - the quote-parity validator - is real and correct, but it did not explain this recurrence, because it was never the cause of it). The true cause was simpler and had nothing to do with text transmission: the owner was sending `/new` (not `/full`) with the real 8-scene, 25-real-quote story every time. `/new`'s own math - `line_count = max(2, round(target_seconds / ESTIMATED_SECONDS_PER_LINE))`, default `target_seconds = 60.0`, `ESTIMATED_SECONDS_PER_LINE = 4.0` - deterministically gives `round(60/4) = 15` on the default duration, which is exactly the "15" that kept recurring, and `_LINES_SYSTEM_PROMPT` (unlike `/full`'s `_COMPLETE_LINES_SYSTEM_PROMPT`) explicitly allows the model to SELECT or invent when the premise has more real quoted moments than `line_count` holds - which is exactly what `/full` exists to prevent. The real, severe consequence: a real $6.838 charge for a wrong video (a woman narrating a story instead of the real cast speaking their own lines), which was then approved and published live to a real public YouTube URL before anyone caught it.

**The real gap, in the owner's own words** ("لماذا داءما تقصير منك؟ ... قل لي كيف اطلبه منك يا حقير؟" - why is it always my shortcoming, tell me how I should even ask you): every previous fix this session checked the *text* sent to the bot (quote parity, cast size, verbatim-copy instructions) and never once checked the *command* itself against what the text actually needs. A real story with more real quotes than a `/new` duration's `line_count` can hold is now refused BEFORE the writer is ever called - zero real cost risked - with a message that states the real mismatch (how many real quotes found vs. how many lines this duration builds) and points the owner at `/full`, or at a longer `/new` duration if selection was genuinely intended.

**Built**: `dialogue_writer.py`'s private `_count_quoted_moments()` was renamed to the public `count_quoted_moments()` so `conversation.py`'s `handle_new()` can call the exact same real counting logic `/full` already uses - no second, divergent implementation to drift out of sync. `handle_new()` now counts the premise's real quoted moments right after computing `line_count` and, if the count exceeds it, refuses with a clear bilingual-structured Arabic message instead of silently building a screenplay that would have to select or invent.

Covered by `tests/test_bot_conversation.py` (`test_new_refuses_when_the_premise_has_more_real_quotes_than_its_line_count_without_calling_the_writer` - 16 real quotes against the default 15-line duration, asserts the refusal names `/full`, and that the underlying mock reasoning client is never exhausted by a real call; `test_new_allows_a_premise_whose_real_quote_count_fits_its_line_count` - exactly as many quotes as lines still proceeds normally). Mutation-tested: reverted the guard block in `conversation.py`, confirmed the new refusal test fails with a real error (the mock reasoning client raises `AssertionError: MockAgentReasoningClient ran out of scripted replies`, proving the writer really was about to be called when it should not have been), restored, confirmed the full 157-test suite green.

**What this does NOT fix**: it cannot undo the real $6.838 already spent or the video already published - that real YouTube URL still needs the owner to make it private or delete it directly, since this code has no access to the owner's own YouTube account to do that for them.

## No narrator, structurally - a code-level gate, not just a prompt request (2026-10-21)

Real, forceful standing instruction ("لماذا تصر ان يوجد الراوي في الكودات يا ابن الحرام؟ احذف الراوي للابد الا بطلب مني" - why do you insist the narrator exists in the code, delete it permanently unless I ask you myself): the fix from 2026-10-19 ("اريد ممثلين يتكلمون لا راوي يروي قصة") only ever told the model, through `_CAST_SYSTEM_PROMPT`/`_LINES_SYSTEM_PROMPT`, not to cast or write a narrator role - a prompt instruction the model can still ignore on any given real call, with no code anywhere checking that it actually complied. The real $6.838 wrong video documented just above - a woman narrating a story instead of real characters speaking - is exactly what that compliance failure looks like in production.

**Built**: `_NARRATOR_NAME_PATTERN` (`راوي|narrator`, case-insensitive) is now checked against every cast member's name inside `_parse_cast()`, BEFORE a screenplay is ever built from it. Any name containing "راوي" (bare, or inside a longer name like "الراوي الأول"/"راوي القصة") or "narrator" (any case) raises `DialogueWriterError` immediately, refusing the whole screenplay rather than silently accepting a narrator character. This is a hard, code-level gate independent of whatever the model decided to do - the narrator prohibition no longer depends on the model reading and obeying a sentence in its own system prompt.

Covered by `tests/test_dialogue_writer.py::test_write_dialogue_screenplay_raises_if_the_model_casts_a_narrator_anyway` - four real cast replies (`"الراوي"`, `"راوي القصة"`, `"Narrator"`, `"The NARRATOR"`), each asserted to raise `DialogueWriterError` matching `"راوي"` and to never reach the lines-writing call (`len(reasoning.calls) == 1`), so no real cost is risked building a screenplay that was always going to be rejected. Mutation-tested: removed the new check from `_parse_cast()`, confirmed the test fails with a real error (the mock reasoning client is exhausted - `AssertionError: MockAgentReasoningClient ran out of scripted replies` - proving the lines step really was reached when the cast should have been rejected first), restored, confirmed the full 157-test suite green.

## The old silent-video/narrator pipeline deleted outright, not just unreachable (2026-10-21)

Real, forceful, repeated standing instruction, escalating across many messages the same day ("لماذا تصر ان يوجد الراوي في الكودات يا ابن الحرام؟ احذف الراوي للابد الا بطلب مني"; "لا راوي موجود في الكود، لا شخص يحكي"; "احذف كل ماهو مرتبط بالماضي الفيديو الصامت و الراوي ... اجعله صنع افلام و بس بشخصيات تتكلم"): the fixes earlier the same day (the `/new` quote-count guard and the hard narrator-name rejection in `_parse_cast`, both documented above) addressed the narrator *reaching a real production*, but did not address the owner's actual, repeatedly-clarified complaint - that a whole narrator-shaped pipeline still physically existed in this codebase at all, reachable or not.

**Confirmed by a real, exhaustive grep-based audit before deleting anything** (not guessed): `WriterAgent` (`src/agents/writer.py`), `DirectorAgent` (`src/agents/director.py`), `EpisodePipeline` (`src/workflows/episode_pipeline.py`), `ComfyUIClient` (`src/integrations/comfyui_client.py`), `RemoteExecClient`/`RealJupyterRemoteExecClient` (`src/integrations/remote_exec_client.py`), and `wan_prompt_builder.py` (the ComfyUI Wan 2.2 T2V graph builder these depended on) were the real "راوي" mechanism: one prose `Screenplay` with no speakers, `DirectorAgent` turning it into silent `Scene`s, `EpisodePipeline` generating them silently and narrating the premise over the result via TTS. `src/bot/conversation.py` and `src/bot/runner.py` - the ONLY code the live Telegram bot actually reaches - were confirmed (via `grep` for real, non-comment usage) to import and call nothing from any of these: `BotConversation.__init__` takes only `DialogueWriterAgent`/`PublishingAgent`, and `_ChatState.produced` is typed strictly `ProducedDialogueEpisode`. `src/app.py`, however, still unconditionally **constructed** every one of them (`WriterAgent(reasoning)`, `DirectorAgent(reasoning)`, `EpisodePipeline(comfyui, ...)`, real or mock `ComfyUIClient`/`RemoteExecClient`) on every single boot, real or test - dead weight that still physically existed as working, narrator-producing code, exactly what the owner meant by "راوي موجود في الكود" even though the live bot could never reach it.

**Deleted outright** (not commented out, not gated behind a flag): `src/agents/writer.py`, `src/agents/director.py`, `src/workflows/episode_pipeline.py`, `src/integrations/comfyui_client.py`, `src/integrations/remote_exec_client.py`, `src/workflows/wan_prompt_builder.py`, and their five dedicated test files. `src/models.py` lost `Screenplay`/`ScreenplayScene` (the narrator's own prose format), `Scene`/`Episode`/`ProducedClip`/`ProducedEpisode` (the silent-clip shapes), and the `CLIP_FRAMES`/`CLIP_FPS`/`CLIP_SECONDS`/`GPU_HOURLY_COST_USD` constants that only ever sized them. `src/agents/publisher.py`'s `PublishingAgent.publish_episode()` dropped its `ProducedEpisode | ProducedDialogueEpisode` union and `isinstance` branch - it now only ever accepts the one real type the live bot produces. `src/app.py`'s `App` dataclass and `build_app()` lost the `comfyui`/`remote_exec`/`writer`/`director`/`pipeline` fields and their construction entirely. `src/config.py`'s `Settings` lost `comfyui_base_url`/`comfyui_configured`, `jupyter_base_url`/`jupyter_password`/`jupyter_configured`, and `azure_tts_voice` (confirmed, by grep, to be read nowhere outside the deleted `EpisodePipeline` wiring - the real `DialoguePipeline` path takes each character's voice from `character_photos.py` instead). `.env.example` and `README.md` were rewritten to match - the README no longer documents a workflow this repository cannot run.

Mutation-testing a deletion is different from mutation-testing an added guard: there is no fix to revert and re-break, since nothing is being guarded - the proof here is instead that NOTHING outside the deleted files still referenced any deleted symbol anywhere. Verified three ways: (1) a repo-wide `grep` for every deleted class/function/constant name turned up only historical comments explaining what was removed and why, never a live import or call; (2) `python -c "import ast; ast.parse(...)"` over every remaining file in `src/` confirmed no syntax error from a half-finished edit; (3) the full test suite - 119/119 passing, down from 157 by exactly the number of tests that belonged to the five deleted test files, with zero new failures or collection errors anywhere else.

**What stays, unchanged**: `/new`, `/full`, `/edit`, `/caption`, `/cancel`, Approve/Publish - the entire real, live bot flow (`DialogueWriterAgent` → `DialoguePipeline` → `PublishingAgent`) is untouched by this commit. This was a deletion of dead code, not a behavior change to anything the owner actually uses.

## No fallback photo of any kind - deleted, not just stopped from being the default (2026-10-21)

Real, forceful, repeated standing instruction, same day as the narrator-pipeline deletion above ("احذف الستة صور و قلت التوليد حسب القصة" - delete the six photos, I told you generation must be per-story; "اي صور جاهزة يا ابن الحرام؟ ... انا لم اطلب هذا بل طلبت صور حسب القصة"): the real, per-story portrait generation built 2026-10-20 (`fal-ai/flux/dev`, see "Real, story-specific character photos replace the fixed six-photo panel" above) was already the PRIMARY path - attempted for every speaking character, every time, whenever `FAL_KEY` was configured - but it still had a fallback: the six generic placeholder illustrations (`character_photos.py`'s `resolve_placeholder_photo`, picked by gender/age_group) were used whenever real generation failed or `FAL_KEY` was missing. The owner's position, stated and restated: no such fallback may exist in the code at all, for any reason - a character either gets a real photo generated from its own story, or its lines fail and are reported, never a stand-in face.

**Deleted outright**: `character_photos.py`'s `resolve_placeholder_photo` (and `_ASSETS_DIR`/`_VALID_GENDERS`/`_VALID_AGE_GROUPS`), `generate_placeholder_avatars.py` (the script that drew the six illustrations), and the six PNG files themselves (`assets/placeholder_characters/`). `Character.reference_image_path` - the field that carried the fallback path from screenplay-write time all the way into production - is removed from `models.py` entirely; `Character` now carries only `appearance` (the real per-story description) and `voice`. `age_group` is removed from the cast-writing step (`_CAST_SYSTEM_PROMPT`, `_parse_cast`) too - once there is no placeholder photo keyed by gender/age_group, nothing in this codebase reads `age_group` for any purpose at all, so asking the model to produce it was dead weight.

**`DialoguePipeline` redesigned around "no usable photo = the line fails", not "no usable photo = raise and abort everything"**: `_resolve_character_photo` now returns `None` (never a fallback path) when there is no configured `image_client`, when `character.appearance` is empty, or when the real `fal-ai/flux/dev` call itself fails - each case with its own real, specific Arabic reason. `produce()`'s pre-pass only ever populates `resolved_photos` for a character whose photo actually resolved; a speaking character missing from that dict has every one of its lines land in `failed_lines` (zero cost - checked before TTS or any fal.ai call) with the real reason attached, rather than ever sending a stand-in face to a real, billed fal.ai call. The old `DialoguePipelineError("no reference_image_path set")` abort-the-whole-episode check is gone along with the field it checked - a missing photo for one character no longer has to take the whole production down with it, consistent with every other failure class this pipeline already survives.

The one place a missing photo is tolerated rather than failing outright is the on-screen caption group image, and only partially: it is built from whichever captioned characters' photos actually resolved, so one captioned character's photo failing does not stop every caption line in the screenplay from rendering (captions cost $0 regardless and are a documented, pre-existing "known limitation" in this README, unrelated to today's change) - but if NONE of the captioned characters have a resolved photo, there is nothing to render on top of, and those caption lines fail too, same as a spoken line would.

**A real risk found and fixed while finishing this**: `Dockerfile`'s `COPY assets ./assets` copied `assets/placeholder_characters/` into the image. With those six files deleted, `assets/` is now an empty directory `git` does not track at all - the next real `docker compose up -d --build` on the droplet would have failed outright on a `COPY` from a path that does not exist in the build context, breaking the real deploy. Caught before pushing (not discovered live on the droplet): removed the `COPY assets ./assets` line entirely, since nothing in this image needs anything from that directory any more.

Covered by `tests/test_dialogue_pipeline.py` (three distinct, separately mutation-tested failure paths - no image client at all, a character with no appearance, and a real `generate_image` call that raises - each asserting the character's line fails with zero cost and fal.ai's own video endpoint is never even called; plus the caption-group partial-resolution behavior, and the case where no captioned character has a photo at all) and `tests/test_dialogue_writer.py`/`tests/test_bot_conversation.py` (cast replies and `Character` construction updated to the new, photo-free shape). Mutation-tested three separate ways (one per failure branch in `_resolve_character_photo`): each branch's `None` return was temporarily changed to a real, existing fallback file path, confirmed the corresponding new test fails with a real assertion (fal.ai's mock client recorded a call that should never have happened), restored, confirmed the full 121-test suite green.

**What this does NOT fix**: a real production run where every speaking character's photo generation genuinely fails (e.g. `FAL_KEY` quota exhausted) now produces a film with zero real clips rather than a film with wrong/generic faces - which is the owner's own explicit tradeoff, not a gap, but it means a `FAL_KEY` outage is a harder stop than it used to be. There is no way around this without reintroducing some form of fallback face, which the owner has now twice explicitly rejected.

## Dead code found by the owner's own reading of the file inventory (2026-10-21)

Real finding: the owner, reading the earlier full-repository inventory given in this session, pointed at the line-count column for `src/integrations/telegram_client.py` ("ما هو 101؟") - a `TelegramClient` interface + `RealTelegramClient`/`MockTelegramClient` wrapper around python-telegram-bot, built early in this project alongside `telegram_listener.py`-style code from nexus-commerce, but never actually wired into the live bot: `src/bot/runner.py` calls `context.bot.send_message`/`send_video`/`answer_callback_query` directly through python-telegram-bot's own `Application`/`ContextTypes`, never through this wrapper. Confirmed by grep before deleting: no import of `telegram_client`/`TelegramClient` anywhere outside the file itself and its own dedicated test. 101 real lines of genuinely dead code, plus its test file, deleted outright.

Mutation-testing does not apply to a pure deletion of unreferenced code - the proof is that nothing else imports it (confirmed by grep) and the full suite is unaffected (117/117, down from 121 by exactly the four tests `test_telegram_client.py` held, zero new failures anywhere else).

## Full code review requested by the owner - three real bugs found and fixed (2026-10-21)

Real instruction, after a session of real financial loss and repeated incidents ("ليست لي بك ثقة بعد ما خسرت كثيرا، الان راجع كل شيء لكي لا تقول لي الخطأ مني" - I don't trust you any more after losing so much, now review everything so you can't tell me later the mistake was mine): a full read-through of every file the live bot actually reaches, looking specifically for correctness bugs - not just the deletions already done today. Found three, each real, each fixed, each mutation-tested separately.

### 1. Arabic titles/names collapsed the character-photo cache key - the worst kind of bug: silent, free, and invisible

`dialogue_pipeline.py`'s `_slug()` (for the screenplay title) and the inline `char_slug` computation (for a character's name) both matched only `[a-z0-9]` - ASCII. Reproduced directly:

```python
>>> slug("جسد بلا روح - قصة حقيقية عن رجل مطلق")
'episode'
>>> char_slug("فوزية"); char_slug("أحمد"); char_slug("الأب")
'character'; 'character'; 'character'
```

Every real title and every real character name in this entire project is Arabic - this project has never once produced a screenplay in any other script. Both slugifiers silently collapsed to the empty string and fell back to their literal default, regardless of what the real title or name actually was. The real photo cache path is `{output_dir}/characters/{slug}_{char_slug}.jpg`, and `_resolve_character_photo` starts with a plain `if os.path.isfile(cache_path): return cache_path, 0.0, None` - a cache HIT, not a failure, so nothing is ever reported anywhere. The consequence: once the very first real Arabic-named character's photo was ever generated in a given `output_dir`, every later character - any name, any gender, any story, forever - would silently reuse that exact same cached file as a "cache hit," at zero cost and with zero error. This is the precise opposite of what today's other fixes (and the 2026-10-20 build before them) were built to guarantee - "صور حسب القصة لا صور ثابتة" - and it was live, undetected, in the very mechanism meant to enforce it.

**Fixed**: `_slug()` rewritten to be Unicode-aware - it keeps any character where Python's `str.isalnum()` is true (which correctly recognizes Arabic letters, unlike the ASCII-only regex class) and collapses runs of everything else to a single underscore. Both the title slug and the character-name slug now go through this one shared function.

Covered by `tests/test_dialogue_pipeline.py::test_produce_gives_different_arabic_named_characters_their_own_cached_photo` - a real Arabic title with two different Arabic-named characters, asserting the image client is actually called twice (not once, with the second silently served from a false cache hit) and that two distinct files land in the cache directory. Mutation-tested: reverted `_slug` to the old ASCII-only regex, confirmed the test fails for exactly the reproduced reason (`len(image_client.calls) == 1`, not 2 - the second character's real generation call never happened), restored, confirmed the full suite green.

### 2. The real-spend cost ceiling was also silently skipping free ($0) caption lines

`produce()`'s hard cost-ceiling check (`MAX_EPISODE_COST_USD`) used to run before the spoken/caption branch of the per-line loop, so a genuinely free on-screen-text line (`/caption`, no TTS, no fal.ai call) got dropped into `skipped_lines` - reported to the owner as "stopped due to the cost ceiling" - whenever an earlier SPOKEN line had already exhausted the ceiling, even though skipping a $0 line protects no real money at all. This directly contradicts the ceiling's own documented purpose ("a hard ceiling on total REAL SPEND").

**Fixed**: the ceiling check now runs AFTER the spoken/caption branch, so only lines that actually cost something are ever gated by it; a caption line is only ever withheld by its own real failure (no photo available for any captioned character), never by unrelated spoken-line spending.

Covered by `tests/test_dialogue_pipeline.py::test_produce_still_renders_a_free_caption_line_after_the_cost_ceiling_is_reached`. Mutation-tested: moved the ceiling check back to its old position, confirmed the test fails (the caption line lands in `skipped_lines` instead of `clips`), restored, confirmed the full suite green.

### 3. The pre-approval cost estimate never included real portrait generation cost

`conversation.py`'s `_format_screenplay()` builds the one cost number the owner actually checks before pressing Approve - and it only ever summed spoken-line video cost (`ESTIMATED_SECONDS_PER_LINE * PRICE_PER_SECOND_USD`). It never added the real per-character portrait generation cost (`fal-ai/flux/dev`, `$0.025/megapixel`, billed once per character appearing in any line) that `DialoguePipeline.produce()` actually spends. The one number built specifically so the owner can decide before committing real money silently understated the real total every single time.

**Fixed**: the estimate now also adds `(number of distinct characters appearing in any line) * PRICE_PER_MEGAPIXEL_USD`.

Covered by `tests/test_bot_conversation.py::test_new_cost_estimate_includes_real_portrait_generation_cost` (new) and the existing `/caption` estimate test, corrected to the same real total. Mutation-tested: removed the portrait-cost addition, confirmed both tests fail with the real understated number (`$0.80` shown instead of the real `$0.85`), restored, confirmed the full suite green.

### What this review covered and did not

Read start to finish: `conversation.py`, `runner.py`, `dialogue_writer.py`, `dialogue_pipeline.py`, `models.py`, `app.py`, `config.py`, every integration client (`fal_video_client.py`, `image_client.py`, `tts_client.py`, `reasoning_client.py`, `publishing_client.py`), `character_photos.py`, `_text_parsing.py`, `main.py`. A repo-wide grep for the same ASCII-only pattern class (`[a-z0-9]`, `[a-zA-Z]`) confirmed no other instance of it exists anywhere in `src/`. This is a code-level review for correctness bugs, not a live production test - the owner's own standing rule in this project's `CLAUDE.md` still applies: nothing here is "proven" until a real run on the real bot produces a real, inspected result.

## Second review pass, same instruction ("راجع مرات و مرات" - review again and again) - three more real findings (2026-10-21)

### 4. A user-facing message still claimed a fallback photo that no longer exists

`conversation.py`'s `format_produced()` had a `portrait_note` that said, verbatim, "(استُخدمت صورة افتراضية بدلاً منها)" - "(a placeholder photo was used instead)" - whenever a character's real portrait generation failed. This sentence was true when it was written (2026-10-20, when a generic fallback really did exist) and became **false** the moment the fallback itself was deleted earlier today, per the owner's own standing instruction - a portrait failure now means that character's lines failed outright (zero cost), not that production quietly continued with a stand-in face. The message was never updated when the behavior it described was deleted - exactly the kind of mismatch the owner caught once already today ("ما هو 101؟"). Fixed: the message now says the real thing that happens - the character's lines were not produced, no cost, no fallback.

Covered by `tests/test_bot_conversation.py::test_format_produced_reports_portrait_generation_failures_by_character_name`, strengthened with a negative assertion (`"استُخدمت صورة افتراضية" not in reply.text`) and a positive one for the real new wording. Mutation-tested: restored the old sentence, confirmed the test fails on the negative assertion, restored the fix, confirmed the full suite green.

### 5. `/new` and `/full` lost real newlines the same way `/edit` used to, before its own 2026-10-20 fix

`runner.py`'s `_new`/`_full` handlers built the premise from `" ".join(context.args)` - python-telegram-bot's own whitespace-tokenized argument list, which treats a real newline exactly like a space. Reproduced directly: `/new 30 line one\nline two\nline three` arrives as `context.args = ["30", "line", "one", "line", "two", "line", "three"]`, and `" ".join(...)` collapses every line break into a single space before the premise ever reaches `handle_new`. This is the identical defect class `_edit` was fixed for on 2026-10-20 ("context.args splits the message on ANY whitespace, including newlines, collapsing multiple edits into one unparseable blob") - left unfixed in the two handlers that build a real story's premise. The dominant real workflow (sending the story as separate plain messages first, then a bare `/new`/`/full`) never triggers this, but a story typed directly after the command on one message does, and silently loses its paragraph structure.

**Fixed**: both handlers now read `update.message.text` directly and split only once (`split(None, 1)`), the exact same fix already proven for `/edit`.

Not independently unit-tested - `runner.py` is explicitly documented (its own module docstring) as needing a real bot token and live Telegram connection to exercise at all, the same reason `/edit`'s identical fix was never unit-tested either. Verified by direct reproduction (shown above) and code symmetry with `/edit`'s already-proven fix, and will be verified live on first real use.

### 6. Two cast members with the same name would silently merge into one

`dialogue_writer.py`'s `_parse_cast()` never checked for duplicate names. `DialoguePipeline.produce()` keys its character lookup with a plain dict comprehension (`characters_by_name = {c.name: c for c in screenplay.characters}`), which keeps only the LAST entry for a repeated key - if the model's cast reply ever named two different characters the same thing (a real, if infrequent, LLM failure mode - two father figures, a reused generic role name, or simple model error), the first one's gender/voice/appearance would be silently discarded, and every line attributed to that name would use the second character's real generated photo and voice. This is the exact "wrong face for a character" defect the owner has fought all session, reachable through a completely different door than the Arabic-slug bug found in the first review pass.

**Fixed**: `_parse_cast()` now checks for duplicate names and raises `DialogueWriterError` naming them, before any screenplay is built or any line-writing call is made.

Covered by `tests/test_dialogue_writer.py::test_parse_cast_rejects_two_characters_with_the_same_name`, asserting the error names the duplicated character and that the lines-writing call is never reached (no real cost risked on a screenplay that was always going to be built wrong). Mutation-tested: removed the check, confirmed the test fails (the mock reasoning client is exhausted trying to serve a lines-writing call that should never have happened), restored, confirmed the full suite green.

## A real, already-paid video could be stranded and lost if only its local duration measurement failed (2026-10-21)

Real, forceful instruction ("لا يجب ان يفشل شيء لاني ادفع مالا عند الفشل" - nothing may fail, because I pay money when it fails): most failure paths in this pipeline already cost $0 (a bad screenplay, a missing photo, a fal.ai job that itself errors or times out - confirmed live earlier this project that a timed-out fal.ai job bills $0.000). But one narrow path did not: once `generate_speech_to_video()` itself succeeds, fal.ai has already billed for real, completed work - the real video bytes are written to disk successfully - and only measuring that file's duration (`_ffprobe_duration`, needed to compute its real cost) was left. If THAT alone failed (a corrupt or partial response from fal.ai, a transient ffprobe error), the old code discarded the clip entirely: the real, paid-for, already-downloaded video file sat on the server, and the owner was told only that the line failed - never given the file, not even a path to it.

**Fixed**: `produce()` now recovers the clip instead of discarding it, in two steps:
1. Try to measure the real video's own duration (unchanged, the normal path).
2. If that fails, try measuring the real audio file instead - it was already written successfully moments earlier, and the video is lip-synced 1:1 to it, so its duration is a real, measured stand-in, not a guess.
3. Only if **both** real files fail to measure does the clip get delivered with `cost_usd=0.0` (a safety-margin understatement, never an invented number) and a clear note telling the owner the real paid file is still attached and to check fal.ai's own dashboard for the exact charge.

In every one of these outcomes the real video file is still delivered to the owner - it is never silently lost again.

Covered by two tests in `tests/test_dialogue_pipeline.py`: `test_produce_falls_back_to_the_audio_duration_when_only_the_video_probe_fails` (video probe fails, audio probe succeeds, real measured cost used) and the updated `test_produce_reports_when_a_line_was_billed_by_fal_ai_before_the_failure` (both probes fail, clip still delivered at `$0.00`, failure note explains why). Mutation-tested: reverted to the old discard-on-probe-failure behavior, confirmed both tests fail for the real reason (`len(produced.clips) == 0` instead of 1 - the real paid file genuinely discarded), restored, confirmed the full 122-test suite green.

## Free conversation and collaborative editing of the screenplay (2026-10-21)

Owner's standing instruction: **"اريد الحديث معه كما اتحدث معك"** - I want to talk to it like I talk to you - clarified to mean a full free conversation with discussion and collaborative editing of the story, not just typed commands.

Before this, the only way to change a draft was the exact command syntax `/edit <n> <text>` / `/caption <n>`, and plain text sent outside those commands did nothing but accumulate silently (`handle_plain_text`, always returns no reply). There was no way to ask a question about the draft, discuss a change in free Arabic, or have the bot explain what it did.

**Added**: `DialogueWriterAgent.route_message()` (`src/agents/dialogue_writer.py`) - a real LLM call that reads the current screenplay draft (if any), whatever text is already accumulated for this chat, and the owner's new message, and returns one structured decision: `accumulate` (still building the premise), `discuss` (answer a question, change nothing), `build_screenplay` (generate the draft - verbatim mode when the owner's words contain quoted dialogue, same guarantee `/full` already gives; estimated mode otherwise, same as `/new`), `edit_lines` (apply exact line edits), or `set_captions` (mark lines as on-screen text, zero cost). `BotConversation.handle_message()` (`src/bot/conversation.py`) dispatches on this decision into the exact same methods `/new`, `/full`, `/edit`, `/caption` already used - no new code path to the real pipeline, no new way to spend money.

**The one real-money call stays exactly where it was.** `handle_message()` and everything it calls has no import of, and no reference to, `DialoguePipeline` at all - confirmed by grep before writing this entry, the same way the dead `telegram_client.py` file was confirmed unused before deleting it. `dialogue_pipeline.produce()` is only ever reached from `_produce_and_notify()` in `runner.py`, which is only ever spawned from the `"✅ موافقة"` branch of the button handler (`_button`, `data == "approve"`). However the owner phrases a message - a question, a correction, a whole new story told conversationally - the real spend still only starts from that one physical button press, unchanged.

**Verbatim-text preservation carries into the router too**: the owner's own standing rule across this whole project - every exact word he writes must be used, nothing dropped or paraphrased - is written directly into `_ROUTER_SYSTEM_PROMPT`: when the owner gives exact quoted words for an edit, `route_message()` is instructed to use them exactly; it only generates new text when the owner describes a change without giving exact words.

**Debounce, because Telegram's own client still splits a long message**: the 2026-10-19 finding above (Telegram's client auto-splits one long typed/pasted message into several separate text updates) means routing each fragment through the LLM individually would ask it to interpret a sentence cut off mid-word. `runner.py`'s `_text` handler now restarts a 1.5-second per-chat timer (`_TEXT_DEBOUNCE_SECONDS`, `_debounce_tasks`) on every fragment; only once no further fragment arrives does `_route_after_debounce` call `handle_message(chat_id, "")`, which pulls the full coalesced text out of `pending_text` itself. The task reference is kept in `_debounce_tasks` until it fires or is cancelled, the same strong-reference pattern `_background_tasks`/`_spawn` already uses in this file for the identical documented asyncio risk (a bare `Task` the event loop holds only a weak reference to can be garbage-collected before it runs).

Covered by 8 new tests in `tests/test_dialogue_writer.py` for `route_message()` (accumulate, edit-with-verbatim-quotes, set_captions, build_screenplay, discuss-never-implies-production, rejects-unrecognized-intent, and two code-level guard tests: rejects edit/caption intents when no screenplay draft exists yet, and rejects an edit_lines reply with no real edits in it) and 10 new tests in `tests/test_bot_conversation.py` for `handle_message()` (accumulate, discuss-never-triggers-production, build_screenplay in both verbatim and estimated mode, build_screenplay using previously-accumulated pending text, edit_lines applying exact quoted words, set_captions, edit_lines refused without a draft, and the empty-text/pending-text consolidation the debounce depends on, both the pull-from-pending case and the nothing-pending-does-nothing case). The no-draft guard and the empty-text/pending-text behavior were specifically mutation-tested (guard removed - test fails because the mock reasoning client is asked for a lines-writing call that should never happen; pending-text pull removed - test fails because the accumulated text is never routed). Full suite: 140/140 passing.

Not independently unit-tested at the `runner.py` wiring layer itself (the file's own long-standing documented limitation - it needs a real bot token and live Telegram connection to exercise at all); verified instead by direct reading of the call chain from `_text` through `_route_after_debounce` to `handle_message`, and will be verified live on first real conversational use, the same way `/new`'s and `/full`'s newline fix above was.

### Real, live incident the first time this was actually used: two separate problems, found from a real screenshot

The owner tested the feature live, minutes after deploy. Two real things happened, both from his own photographed phone screen - not guessed:

**1. A redeploy triggered to fetch logs wiped his in-progress conversation.** `BotConversation._states` (the dict holding `pending_text`/`screenplay` per chat) lives only in the running process's memory - it was never persisted, in this feature or any earlier one. After confirming the first deploy succeeded, a second `workflow_dispatch` run was triggered purely to pull a fresh `docker compose logs` tail - which recreates the container (`docker compose up -d --build`) and therefore resets every chat's in-memory state. The owner's new free-text message ("اريد ان تصنع القصة كما اكتبها منك بالظبط لا تغيير فيها" - make the story exactly as I wrote it, no changes) landed right after that reset, with nothing left in `pending_text` to build from - so the premise sent to the LLM was the bare instruction sentence alone, with no actual story in it, and the real cast-extraction call correctly came back with zero characters to extract.

Nothing here is a code defect to fix - it is a real operational mistake: redeploying for diagnostic convenience while the owner was actively mid-conversation on the live bot. **The fix is procedural, not code**: don't redeploy to fetch logs while a real conversation may be in flight - read code and existing logs first, and treat any live redeploy during active use as something that resets every open conversation, stated plainly beforehand, not discovered after the fact from a confused owner.

**2. When the cast extraction genuinely failed, the raw internal exception text reached the owner verbatim**: `Cast reply was not a non-empty JSON array: '[]'` - English, a Python `repr()` of the raw model reply, no indication of what to do next. This matches an existing pattern already present in `/new`'s own `count_quoted_moments` error handling (`return BotReply(text=str(exc))`) - not a regression introduced today, but the first time a real owner actually hit it through this new path and the first time it was actually seen live. Left as-is for now (changing the message text itself is a pure wording fix, not a safety fix, and is intentionally not bundled into the same change as the real fix below - the owner's own instruction today was explicit: any change risks more real money, so this round makes the smallest real fix and nothing else).

**The one real code defect found and fixed**: `_route_after_debounce` in `runner.py` called `conversation.handle_message(chat_id, "")` with no error handling at all. `handle_message()` only catches `DialogueWriterError` - a real reasoning-client failure (a genuine network error, or any other raw exception `RealGroqReasoningClient`/`RealGeminiReasoningClient` can raise on a real API failure, never wrapped as `DialogueWriterError`) would propagate straight out of this bare `asyncio.create_task`. Unlike the other command handlers, this task is never dispatched through python-telegram-bot's own `Application`, so the global `add_error_handler` registered below (which exists specifically to report any uncaught exception from a real handler to its chat) never sees it - the exact same structural gap `_produce_and_notify` was fixed for on 2026-10-19, reopened here through the new debounce path. A real failure here would have died as an unretrieved `Task` exception: total silence to the owner, the one failure class this whole project exists to prevent.

**Fixed**: wrapped the `handle_message()` call in `_route_after_debounce` with a broad `except Exception`, sending the owner a real (if technical) error message instead of nothing. Full suite re-run after the fix: 140/140 (the fix touches only `runner.py`, which this project's own tests cannot reach directly - verified by re-reading the full call chain, same as the debounce wiring itself above).

## Owner's instruction after the live incident ("تثبت و دقق و اصلح كل شيء" - verify, scrutinize, fix everything): a deeper review pass of the new feature and the production pipeline, two more real bugs found (2026-10-21)

### 7. The owner's real accumulated story text was silently destroyed whenever the router answered "discuss", "edit_lines" or "set_captions"

`handle_message()`'s debounce-pulled path (`text == ""`, called by `_route_after_debounce`) pulled the full accumulated `pending_text` into `text` and cleared `state.pending_text = ""` **unconditionally**, before even knowing what the router would decide. Only the `"accumulate"` branch (re-stores it via `handle_plain_text`) and `"build_screenplay"` (consumes it into a screenplay) ever put anything back. The real, live incident earlier today was exactly this: a message got classified as something other than `"accumulate"`/`"build_screenplay"`, and whatever real story text the owner had already typed - potentially a long, real, personal story, accumulated across several Telegram-split messages - was gone, forever, the moment that happened. No error, no warning - it simply stopped existing. This is the direct opposite of the owner's own most-repeated standing rule in this entire project: every letter and word he writes must be used, never dropped.

**Fixed**: every branch that does not itself consume the pulled text (`discuss`, `edit_lines`, `set_captions`, and the structurally-unreachable fallback) now restores `state.pending_text = text` before returning, so a router classification that isn't "build it now" or "still more story" never costs the owner anything real.

Covered by three new tests in `tests/test_bot_conversation.py` (`test_handle_message_discuss_preserves_pending_text_pulled_from_debounce`, `..._edit_lines_preserves_unrelated_pending_text...`, `..._set_captions_preserves_unrelated_pending_text...`), each proving the text survives a SECOND debounce-pulled call (which would otherwise return empty immediately) and is never duplicated (the router sees it exactly once). Mutation-tested: removed all three restore lines, confirmed all three tests fail for the real reason (`assert '' == 'تم.'` - the real text genuinely gone), restored, confirmed the full suite green (143/143 at that point).

### 8. A retried production could silently show the wrong characters in a caption frame

`DialoguePipeline._build_caption_group_image()`'s cache path was keyed only by the screenplay's own title-derived `slug` - the exact same slug per-character photos are deliberately, correctly cached by (a real, intentional retry of the same screenplay after an `/edit` must reuse an already-generated character photo, not regenerate and re-bill it). But the caption GROUP photo's cache key carried no trace of which characters were actually in it. A retry of the same screenplay (same title, same slug) after an `/edit` or `/caption` that changed which characters are captioned - a real, ordinary workflow this project explicitly supports - would silently reuse the group photo built for the FIRST, now-stale caption roster: the wrong character's face shown on screen during a line that character never even appears in, with no error anywhere to catch it.

**Fixed**: the cache filename now also encodes every included character's own name (through the same `_slug()` the per-character cache already uses), so a genuinely different caption roster under the same screenplay title can never collide with a stale file.

Covered by `tests/test_dialogue_pipeline.py::test_produce_does_not_reuse_a_stale_caption_group_photo_after_the_caption_roster_changes` - two real `produce()` runs under the identical title but with different single-character caption rosters, asserting two distinct group-photo files exist on disk afterward (not one reused/stale file). Mutation-tested: reverted the cache key to the old title-only filename, confirmed the test fails for the real reason (`assert 1 == 2` - only one file, the second run silently reused the first), restored, confirmed the full suite green (144/144).

### 9. A second real, live hit on the same owner: a raw English model reply reached him as a confusing "error"

Minutes after the fixes above shipped, the owner tried again - sent only the instruction "اريد ان تصنع القصة كما اطلبها منك بالظبط لا تغيير فيها" (make the story exactly as I asked, no changes), with no real story text accumulated (the earlier incident's container restart had already lost it, and this message carries no story content of its own - an instruction, not a premise). The casting model, given a premise with nothing to cast, did not error - it replied in plain English asking a clarifying question: `"What is the story premise or the characters you would like included?"`. `_parse_cast()`'s JSON-decode failure put that raw reply straight into the owner's chat as `Cast reply was not valid JSON: 'What is the story premise...'` - confusing, in the wrong language, and told him nothing about what to actually do. The same class of problem as finding #2 earlier in this same file, now hit live a second time by the same real owner within the hour.

**Fixed**: both `_parse_cast()` failure paths (invalid JSON, and valid-but-empty JSON array) now raise a clear, actionable Arabic message - "the text you sent has no real story or dialogue to cast characters from; send the complete story as a plain message first, then ask to build the screenplay" - with the model's raw reply kept at the end for anyone debugging, not as the whole message.

Covered by two new tests in `tests/test_dialogue_writer.py` (`test_parse_cast_gives_a_clear_arabic_message_when_the_model_replies_with_no_real_json`, `..._for_an_empty_cast_array`), plus an existing test (`test_write_dialogue_screenplay_raises_on_invalid_cast_json`) updated to match the new message - it was asserting on the old English text, which is exactly the kind of test that silently goes stale when a user-facing message changes for a real reason. Mutation-tested: reverted both messages to their old raw-repr form, confirmed the two new tests fail for the real reason (asserting on Arabic text that was no longer there), restored, confirmed the full suite green (146/146).

**This does not fix the owner's actual blocker** - resending only the instruction, with no story text behind it, will always fail to produce a cast, in any language the error is shown in. The real fix for that is procedural, already said directly to him: resend the complete story itself (not just the instruction) after any restart, since nothing survives one yet (see finding #1 above on `pending_text` losing nothing it already has - which does not help a restart that happened before this message was ever typed).

### 10. A real, live money-adjacent incident: pressing ✅ Approve itself failed, because a router call had frozen the whole bot

The owner successfully built a real screenplay (a real $6.05 estimate shown, buttons offered), pressed **✅ موافقة**, and got back a real Telegram API error instead of production starting: `Query is too old and response timeout expired or query id is invalid`. No production ever started - `query.answer()` is the first line of `_button` in `runner.py`, so the handler aborted before reaching `handle_approve()` or `_produce_and_notify()` at all. Zero real money was at risk here, but the owner's own approval - the one explicit, deliberate action this entire project's safety model is built around - was silently swallowed, with a cryptic Telegram-internal error instead of a clear explanation or an automatic retry.

**Root cause**: `_route_after_debounce()` called `conversation.handle_message()` **directly on the event loop** - a real, synchronous HTTP call to the reasoning client (`RealGroqReasoningClient`, falling back to `RealGeminiReasoningClient` on failure, whose own retry path calls real `time.sleep()` for up to 20 seconds per attempt, up to 3 attempts - a genuine ~60-second blocking call in the worst case). `_new` and `_full` in `runner.py` had the exact same defect, calling `conversation.handle_new()`/`handle_full()` (each making two real reasoning-client calls) directly on the event loop too - a pre-existing gap, just never caught live before today, because `/new`/`/full` fire far less often than the new router fires on every debounced free-text message. **While any of these calls is in flight, the bot's single event loop cannot do anything else at all** - not answer a callback query, not process another chat's message, nothing. The owner pressed Approve while a router call (from an earlier free-text message) was still blocking the loop, and by the time the loop was free again, Telegram had already invalidated that callback query.

This is the identical class of bug `DialoguePipeline.produce()` was already deliberately protected against (`_produce_and_notify` runs it `loop.run_in_executor(...)`, with the module's own docstring explaining exactly why: "so /new, /edit and button presses on OTHER chats keep working while a generation is in flight") - but that protection was only ever applied to the pipeline's own slow call, never to the reasoning-client calls underneath `handle_new`/`handle_full`/`handle_message`, which are real, synchronous, and can individually take tens of seconds.

**Fixed**: all three - `_new`, `_full`, and `_route_after_debounce` - now run their `conversation.*` call through `loop.run_in_executor(None, ...)`, the same proven pattern, so a slow or retrying reasoning-client call never again freezes button presses or other chats.

Not independently unit-tested (this is pure `runner.py` event-loop wiring, which this project's tests cannot reach directly, per the file's own long-documented limitation); verified by re-reading the exact call chain and confirming it now matches `_produce_and_notify`'s already-proven pattern exactly. `python -m py_compile` confirms the file is still syntactically valid; the full test suite (146/146, unaffected - none of it touches `runner.py`) stays green.

### 11. The owner's own real question ("و اين groq؟ اليس واحد منهم احتياطي؟") exposed a real visibility gap

Groq is the real primary reasoning provider, Gemini the real fallback (`src/app.py`: `RealGroqReasoningClient(..., fallback=gemini_fallback)`, only reached when `GROQ_API_KEY` is configured) - this was already true, and already correct. But `RealGroqReasoningClient.complete()`'s `except Exception:` block discarded the real Groq failure reason completely before calling the fallback - not into a message to the owner (reasonable; the fallback is meant to be invisible when it works), but **not even into the container's own logs**. When the owner asked a direct, reasonable question - did Groq fail, and why, before Gemini's own real 500 reached him - neither of us could answer it from anything the system had kept. A previously-documented real failure mode of this exact pair (`docs/SOURCES.md` "Groq org-wide quota" - a second Groq key sharing another project's daily token quota) could be happening again right now and would be just as invisible.

**Fixed**: the real exception is now logged (`logger.warning`, `nexus_studio.reasoning_client`) before falling back to Gemini, and also before re-raising when no fallback is configured - so the next time this happens, `docker compose logs` has the real answer instead of a shrug.

Covered by two new tests in `tests/test_reasoning_client.py` (`test_groq_complete_logs_the_real_failure_reason_before_falling_back`, `..._with_no_fallback_too`), using pytest's `caplog` to assert the real exception text reaches the log at WARNING level in both cases. Mutation-tested: reverted both `logger.warning` calls, confirmed both tests fail for the real reason (no log record at all), restored, confirmed the full suite green (148/148).

### 12. Answering the owner's own follow-up directly, instead of sending him to check it himself ("افتح انت و تثبت انت و دقق انت")

The owner asked whether `GROQ_API_KEY`/`GEMINI_API_KEY` are really configured, after finding only the three deploy secrets (`DEPLOY_HOST`/`DEPLOY_USER`/`DEPLOY_SSH_KEY`) in GitHub's own repository secrets page. Those are real, but irrelevant to the question - `.env` (where every real API key actually lives) is gitignored and only ever exists on the real droplet, never in GitHub, never readable from a coding session with no SSH access to that server. Told to check it himself first; he pushed back and asked for it to be checked directly instead.

`.github/workflows/deploy.yml` (the only real SSH access this session has to the droplet at all) now appends a real, read-only check after every deploy: for each of the five keys the app actually uses (`GROQ_API_KEY`, `GEMINI_API_KEY`, `FAL_KEY`, `AZURE_SPEECH_KEY`, `TELEGRAM_BOT_TOKEN`), it prints `SET` or `EMPTY or MISSING` - **never the real value itself**, only whether the line in `.env` has anything after the `=`. This answers the real question from every future deploy's own log, not just this one time, and the same real mechanism now also shows directly whether Groq is configured at all (making finding #11 above fully actionable: if Groq shows `EMPTY or MISSING`, Gemini is running alone as the real primary, not as a fallback, and every Gemini failure is the whole story, not half of it).

No Python code changed, no tests apply (pure deploy-workflow shell). Verified the only real way this document's rules allow: triggered a real deploy run and read its real output.

### 13. The real spend ceiling can be higher than a real fal.ai balance

The owner's real, live concern: his real fal.ai account balance is under $8, and `config.py`'s `MAX_EPISODE_COST_USD` - the hard real-spend ceiling `DialoguePipeline.produce()` stops at - defaults to **$10.0** whenever it is not set in `.env`, which `.env.example` does not even list as a variable to fill in. If that default is in effect on the real server, the ceiling would let real spend climb past a real balance that is genuinely lower than it, before fal.ai's own account balance check (not this code) would be what actually stops it.

The deploy workflow's diagnostic step (finding #12) now also prints the real, live `MAX_EPISODE_COST_USD` value from `.env` - unmasked, since it is a number, not a credential - or states plainly that config.py's $10.0 default is in effect if the line is missing. This lets the owner (or a future deploy's own log) compare the real ceiling against his real account balance before ever pressing Approve, instead of finding out from an incomplete production run.

## The real root cause of the whole evening's confusion: free text was racing the owner's own /full and /new commands (2026-10-21)

After hours of confusing, seemingly-random failures tonight (a screenplay about "where is the file" that the owner never asked for; `/full` repeatedly replying with its own usage message immediately after he had just sent a real story; a real quote count that seemed to change between sends), the real mechanism was found by tracing the exact call path, not guessed: in the real deployed bot, `BotConversation.handle_message()` is **only ever called from `runner.py`'s debounce timer, with an empty `text`** - `_text()` never calls it with real text directly, it only accumulates (`handle_plain_text`) and (re)starts a 1.5-second debounce. That means every single real call to `handle_message()` in production has `pulled_from_pending = True`.

`handle_message()`'s `"build_screenplay"` branch did not distinguish this from a genuinely real-time, explicit request - if the router classified the accumulated text as ready to build (which a finished-looking story, even without an explicit "build it now" instruction, can plausibly look like to the model), it would **silently build the screenplay and consume `pending_text`, roughly 1.5 seconds after the owner went quiet** - racing whatever he typed next. If his own `/full` or `/new` landed after that automatic build had already fired, it found nothing left to build from (the real "الاستخدام: ..." usage message, which looks exactly like an error to someone who had just sent a real story seconds earlier). Two real, different screenplays being built from the same evening's typing, a story that looked like it had a different quote count between sends, and `/full` repeatedly finding "nothing" right after a real story was sent - all of these are explained by this one race, not by several unrelated bugs.

**Fixed**: a debounce-driven call (`pulled_from_pending = True`) now **never** auto-builds a screenplay. The router is still asked, and if it says `"build_screenplay"`, the accumulated text is restored (never lost) and the reply explicitly tells the owner to send `/full` or `/new` himself. Building a screenplay through free conversation now only happens for a genuinely real-time, non-empty message passed directly to `handle_message()` (unreachable from the live bot's own `_text` handler as it is wired today, but kept for a direct caller / future wiring, and covered by the existing tests that pass real text explicitly). `/full` and `/new` are now the only way the live bot ever builds a screenplay - predictable, with no timing window where typing fast or slow changes what happens.

Covered by `tests/test_bot_conversation.py::test_handle_message_never_auto_builds_a_screenplay_from_the_debounce_path`: proves a debounce-driven "build_screenplay" classification does not build anything (`handle_approve` still has nothing to approve), and that the real story text survives intact for an immediate, explicit `/full` right after (its real content, `"مرحبا"`, reaches the built screenplay). Mutation-tested: removed the `pulled_from_pending` guard, confirmed the test fails for the real reason (a screenplay WAS auto-built, `assert "/full" in reply.text` failed against the real built-screenplay text instead), restored, confirmed the full suite green (149/149).

### The same race, still there - because the real bug was never "what intent gets auto-built", it was "there is a window where pending_text is empty at all" (2026-10-21)

Minutes after the fix above shipped, the owner hit the identical symptom again: sent his real story, got the real "22 quoted moments" warning from `/new` (a sign he had explicitly sent `/new`, not shown in the screenshot but necessarily true - that warning only exists in `handle_new()`), then sent `/full` immediately after and got its usage message back - "nothing to build" - seconds after sending a real story.

The fix above was real and necessary, but incomplete: it stopped the debounce path from **deciding to build**, but `handle_message()` still **cleared `state.pending_text` to `""` the moment it was pulled**, intending to restore it in every branch that does not consume it - including the new deferral reply added above. That restore only happens **after** `route_message()` returns - a genuine, slow, synchronous network call, now correctly run on its own executor thread (see the event-loop-freeze fix above) specifically so it does not block the rest of the bot. That is exactly the problem: running it off the main thread means the owner's own `/full` or `/new` - now *also* running on its own executor thread, from that same fix - can genuinely execute concurrently with the debounce handler, and can observe `state.pending_text` sitting at `""` during the real window between the clear and the restore. The second fix (never auto-building) made that window *survivable* in most cases by restoring the text a moment later - but a `/full` or `/new` that happened to run during that exact window still saw nothing, no matter how fast the restore that followed.

**Fixed properly this time**: `handle_message()` no longer clears `state.pending_text` speculatively at all. It only ever *reads* from it (`source_text = state.pending_text if pulled_from_pending else text`), and only ever *mutates* it at the one real point something actually consumes it - a successful screenplay build. Every other branch (`accumulate`, `discuss`, `edit_lines`, `set_captions`, the build-deferral reply, the unreachable fallback) now reads but never writes it, so there is no window of any length, however short, where a concurrent read from `/full`, `/new`, or anything else can observe it as empty when it is not.

The existing tests already enforce this precisely: mutation-tested by reintroducing a one-line "clear pending_text once pulled" (the exact shape of the original bug, re-added deliberately to check), confirmed 4 real tests fail for the real reason (`assert '' == 'تم.'` / `assert "/full" in reply.text` against a real, already-built screenplay) - not a synthetic concurrency test, but the same black-box behavior tests already in place, which is exactly what "never observably empty" means from the outside. Restored, confirmed the full suite green (149/149) with no new tests needed - the existing coverage was already strong enough to catch this specific regression; what changed is the implementation guaranteeing it structurally rather than by remembering to restore in every branch.

## Owner's instruction, "راجع كل شيء" (review everything) - one more pass over runner.py's new concurrency surface (2026-10-21)

The three fixes above closed the real data-loss and freeze bugs, but `runner.py` itself still had two real, if lower-severity, loose ends from the same evening's work:

**1. Explicit commands never cancelled the pending debounce task.** `/new`, `/full`, `/edit` and `/caption` could still have a debounce task armed from an earlier text fragment, which would fire a few moments after the command's own reply - usually finding nothing to do (harmless, thanks to the fix above), but if the command was still mid-flight on its own executor thread, the debounce task could run concurrently, make its own redundant real reasoning-client call, and send a second, confusing message right after the owner had just gotten what they asked for. `_cancel_debounce()` now runs at the start of every explicit command and the button handler - the same "a newer, explicit action supersedes the automatic one" reasoning `_text()` itself already applies to a newer fragment cancelling an older one, just applied to an explicit command superseding the debounce path entirely.

**2. The publish button's real YouTube upload blocked the event loop.** `handle_publish()` → `PublishingAgent.publish_episode()` → `client.upload_video()` is a real, synchronous HTTP upload of the real stitched video file - called directly from `_button`, exactly the same freeze class as the `/new`/`/full`/debounce bug fixed earlier tonight, just not yet hit live (a video upload is slower and rarer than a text reasoning call, but no smaller a block while it runs - every other chat, every other button, frozen for the whole upload). Fixed the same way: `loop.run_in_executor(None, conversation.handle_publish, chat_id, platform)`.

Neither is independently unit-tested (pure `runner.py` wiring, the file's own long-documented limitation); verified by `python -m py_compile` and re-reading the call chain against the now-established, already-proven pattern. Full suite: 149/149, unaffected.

## Owner's explicit instruction, once the real race above was actually fixed: bring auto-build back ("السيناريو تلقائيا مع الاختيار يبقى هو لي") (2026-10-21)

Disabling auto-build from the debounce path (two sections up) was the right call at the time - it was a real, live bug and needed a fix immediately. But it was always a workaround for the race, not what the owner actually wanted long-term: he asked, explicitly, for free conversation to build the screenplay on its own once he goes quiet - "ابن السيناريو تلقائيًا، لكن الاختيار يبقى لي" (build it automatically, but the choice stays mine) - now that the real race (the observably-empty `pending_text` window) is fixed structurally, and `runner.py` cancels any still-pending debounce task the instant an explicit command arrives (so an explicit `/full` or `/new` always wins outright, never just races the automatic path).

**Re-enabled**: the debounce-driven path (`pulled_from_pending`) now builds a screenplay exactly like a direct call does, when the router says `"build_screenplay"`.

**A real bug was caught immediately by the test for this exact change, before it ever reached the owner**: the premise-building line still unconditionally combined `state.pending_text` with the newly-read text - but when `pulled_from_pending`, those are now **the same text** (per the earlier fix, `source_text` is read directly from `state.pending_text`, never cleared). Concatenating them duplicated the owner's own words, which doubled every real quoted moment `count_quoted_moments` found and asked the model for twice as many lines as the real premise actually has. Fixed: when `pulled_from_pending`, the premise is `source_text` alone; the combine-with-`state.pending_text` form only applies to a direct, non-empty-text call, where they are genuinely separate.

`tests/test_bot_conversation.py::test_handle_message_builds_a_screenplay_from_the_debounce_path_once_safe` replaces the earlier "never auto-builds" test (that test's own premise - that auto-building is unsafe - is no longer true, now that the real bug behind it is fixed) and covers both: a screenplay IS built from the debounce path, and - the regression this change itself introduced and the test caught on the first real run - the real premise text is used once, not duplicated. Mutation-tested: reverted the premise fix, confirmed the test fails for the real reason (`"Reply returned 1 lines, needed 2"` - the mock lines reply doesn't match the doubled quote count), restored, confirmed the full suite green (149/149).

## fal.ai billed every real speech-to-video call at the 720p rate, despite every request asking for 480p (2026-10-21)

The owner's real fal.ai account balance went from $14.34 to **-$0.73** overnight, with no new video ever reaching him. His own Usage & Billing page (screenshotted directly) showed, for `fal-ai/wan/v2.2-14b/speech-to-video`: Quantity **117.91**, Unit Type "Seconds", Unit Price **$0.20**, Total Cost **$23.58125** - plus $0.15 for portraits, for a billing-period Total Cost of $23.73. Two real invoices in the same period: Oct 19 ($10.00, paid), Oct 20 ($13.00, paid) - $23.00 paid against a $23.73 real total, hence the negative balance.

$0.20/video-second is fal.ai's own documented **720p** rate (verified live again this session via `WebFetch` against both `https://fal.ai/models/fal-ai/wan/v2.2-14b/speech-to-video/api` and the main model page - "$0.20 per video second for 720p, $0.15 per video second for 580p, $0.10 per video second for 480p", unchanged from the 2026-10-18 verification already in this file). $0.20 × 117.91 = $23.582, matching the real invoice almost to the cent - this is not a rounding coincidence, it is the 720p tier.

**This project's code has never requested 720p.** Checked every way a silent override could exist, and found none:
- `generate_speech_to_video()`'s real request body (`fal_video_client.py`) has sent `"resolution": "480p"` as its own default since the model was first integrated - confirmed via `git log --follow` + `git show` on the very first commit (`0b80a95`) that added this file.
- `DialoguePipeline.__init__`'s `resolution: str = "480p"` default is never overridden anywhere in `app.py` or `config.py` (grepped the whole tree for `resolution=`) - production always constructs the pipeline with the default.
- No retry, fallback, or quality-escalation path anywhere in `dialogue_pipeline.py` or `fal_video_client.py` ever changes `resolution` between attempts.
- Re-verified live (`WebFetch`, this session) that `resolution` really is the parameter's exact name (not `video_size`/`quality`/anything else), that `480p` really is a valid, even-default enum value, and that the endpoint has no separate per-resolution sub-ids that `fal-ai/wan/v2.2-14b/speech-to-video` could be silently resolving to instead.
- A single row in the billing table at one unit price (not several rows at mixed prices) means this was not an occasional mix-up on one call - every real call across at least Oct 19-21 was billed this way.

**Root cause not establishable from here.** Nothing in this codebase, its history, or fal.ai's own published API/pricing docs explains why a request that says `"resolution": "480p"` is billed at the 720p rate - that can only be confirmed from fal.ai's own side (their Activity/Request History pages show what *they* recorded receiving for a specific request id, which only the account owner can read, or their support team directly). **Said to the owner plainly, not guessed at or hidden**: this is very likely a fal.ai-side billing or backend discrepancy, not a bug in this project's request code, and the real next step - opening a fal.ai support ticket with this exact evidence - is his to take, since it requires his account.

**Fixed what the code *can* fix immediately: stop the owner's own real-money safeguards from trusting a rate that has never once matched reality.** Both of the places this project tracks real spend against fal.ai pricing - the pre-Approve cost estimate (`conversation.py`'s `_format_screenplay`) and the hard `MAX_EPISODE_COST_USD` ceiling (`dialogue_pipeline.py`'s `produce()`) - used `PRICE_PER_SECOND_USD["480p"]` ($0.10/s), the documented-but-never-once-true rate. A ceiling computed at $0.10/s is not a real $10 ceiling when reality is $0.20/s - it is a real $20 ceiling wearing a $10 label, and the pre-Approve estimate shown to the owner before every single decision has been understating his real bill by roughly half, every time, since this feature shipped. Given the owner's standing, repeated instruction ("never may this silently cost him money"), guessing the lower, hopeful number was never defensible once real, repeated invoice evidence said otherwise.

Added `REAL_BILLED_PRICE_PER_SECOND_USD` (`fal_video_client.py`) = `PRICE_PER_SECOND_USD["720p"]`, documented with this exact evidence, and switched both real-money tracking sites to it. **The actual API request is unchanged** - still `resolution="480p"`, still the cheaper, valid, documented choice; there is no evidence that requesting anything else would change what fal.ai actually does here, so changing the request itself would be a guess, not a fix. This is a conservative safety patch pending the owner's dispute with fal.ai, not a claim that the root cause is understood or resolved.

Covered by the existing cost-assertion tests in `tests/test_dialogue_pipeline.py` (9 of them) and `tests/test_bot_conversation.py` (2 of them), all updated from the `0.10`/`PRICE_PER_SECOND_USD["480p"]` literal to `REAL_BILLED_PRICE_PER_SECOND_USD`. Mutation-tested: reverted both real-money call sites back to the old 480p-rate constant, confirmed all 9 real-cost tests fail for the real reason (e.g. `Obtained: 0.225, Expected: 0.425` - exactly half, as predicted), restored, confirmed the full suite green (149/149).

**Direct confirmation, same evening, no login required:** the owner asked how to give Claude access to his fal.ai account to settle this definitively. He should never hand over a login - fal.ai has its own documented REST API (`GET /v1/models/requests/by-endpoint`, `expand=payloads`) that returns `json_input`, the ACTUAL request body fal.ai recorded receiving for each past request, authenticated with the same `FAL_KEY` already in `.env` - no new credential. Queried it for the last 5 real `fal-ai/wan/v2.2-14b/speech-to-video` calls via a temporary, read-only deploy-workflow diagnostic (removed once read, commits `78e1dac`/`d5b07a4`): all 5 came back `"resolution":"480p"` with `status_code 200`. This upgrades the earlier invoice-math inference to direct confirmation: fal.ai's own system acknowledges receiving a correct, cheap 480p request, and still billed the 720p rate. Definitively fal.ai's side to fix, not this project's.

## A real, severe incident traced: real money spent, zero notification ever reached the owner (2026-10-21)

The owner reported his fal.ai balance dropped by ~$15 overnight with no video ever arriving - not a partial result, not a truncated one, nothing at all ("لم يصلني شيء... تسحب الرصيد بدون اي فيديو"). Cross-referenced against fal.ai's own Request History API (the same one used for the 480p/720p finding above): the real jobs behind that spend show `status_code 200` - they succeeded on fal.ai's side. So the money was spent on real, successfully-generated video - and something in this bot's own delivery path lost it completely, silently.

A dedicated full audit of `runner.py` (the owner's explicit demand: "صلح كل شيء قبل البدء مع replicate لانك لن تجد اعذار للاصلاح" - fix everything before even considering alternatives, no more excuses) found FOUR real, still-live variants of exactly this failure shape, all sharing the same root cause: a bare `asyncio.create_task()` (`_produce_and_notify`, `_route_after_debounce`) is not dispatched through python-telegram-bot's `Application`, so `add_error_handler`'s global catch never sees an exception raised inside one - and several of the EXISTING failure-reporting `send_message()` calls inside these tasks were themselves unprotected. If the notification call itself failed (an ordinary, real Telegram rate-limit or transient network error - exactly the kind of thing that happens on a real, cheap VPS over a real night of heavy use), the exception died as an unretrieved Task exception, and the owner got the exact symptom he reported: real money spent, real video generated, and total silence on his phone.

1. **`_produce_and_notify`'s own two failure-reporting `send_message()` calls were unprotected** (the function the owner's money flows through) - the single most likely mechanism behind the reported incident, since it sits directly after the real fal.ai spend.
2. **`_route_after_debounce`'s final `_send_reply()` call was outside its own try/except** - same shape, no money attached, but the owner would see the identical "I typed something and got total silence" symptom.
3. **`_on_error` (the Application's own global, last-resort error handler, reachable from every `/new`, `/full`, `/edit`, `/caption`, and the approve/publish buttons) swallowed a failure of its own notification with a bare `except Exception: pass`** - not even logged. The one place meant to be the final safety net had no safety net of its own.
4. **A secondary, non-silence correctness bug in `_send_reply`'s clip-sending loop**: a failure on clip N used to abort the loop and let `_produce_and_notify` report a blanket "فشل إرسال الفيديو إليك" (the video failed to reach you) even when clips before N had already sent successfully and were genuinely sitting in the owner's chat - a real, confusing mismatch between what the bot claims and what the owner's own chat shows.

**Fixed:** added `_notify_or_log()` - every failure-reporting `send_message()` in the file now goes through it; if the notification itself fails, the failure is logged (`nexus_studio.runner`, the same visibility pattern as the Groq-failure-logging fix) instead of vanishing with zero trace anywhere. `_route_after_debounce`'s final `_send_reply()` call is now wrapped in its own try/except. `_on_error`'s silent `pass` now logs. `_send_reply`'s clip-sending loop now attempts every clip regardless of an earlier failure and raises one aggregate, per-clip-named error only if something actually failed - never a blanket claim that contradicts clips the owner already received.

Not independently unit-tested - pure `runner.py` event-loop/Telegram-delivery wiring, this file's own long-documented limitation (it needs a real bot token and live connection to exercise at all). Verified by `python -m py_compile` and by re-reading every new call chain against the now-established pattern (`_notify_or_log` mirrors `reasoning_client.py`'s Groq-failure-visibility fix exactly). Full suite: 149/149, unaffected - none of it touches `runner.py`.

**This does not retroactively explain with certainty which exact historical charge died this way** - there is no log correlating one specific past run to one specific failure point, because the whole problem was that nothing was logged. What it does provide: every one of the four concrete, reachable "real money spent (or real work done), zero notification" gaps found in a full audit of this file is now closed, and any repeat of total silence after this point is a new, distinct regression to find immediately - not a recurrence of an already-known, already-open gap.

## A second real video provider: Replicate, after fal.ai's own billing could not be trusted (2026-10-21)

After the 720p/480p billing finding above, the owner asked directly whether a cheaper, more trustworthy alternative existed, then ran his own real, paid test before committing to anything - exactly the standard this project holds itself to.

**Real verification, not guessed:** `wan-video/wan-2.2-s2v` on Replicate is the same real model family (audio-driven lip-sync from one photo + one audio clip) confirmed live via Replicate's own real API docs (`https://replicate.com/docs/reference/http`) and the real Playground form (owner's own screenshots): endpoint `POST https://api.replicate.com/v1/models/wan-video/wan-2.2-s2v/predictions`, `Authorization: Bearer $REPLICATE_API_TOKEN`, `input: {prompt, image, audio, num_frames_per_chunk, seed, interpolate}` - **no resolution parameter at all**, unlike fal.ai's three tiers. Files accept data URIs directly (Replicate's own docs: "Files should be passed as HTTP URLs or data URLs"), same pattern as the existing fal.ai client.

**The real trust test, run by the owner himself, 2026-10-21:** $10 real credit purchased (invoice `HJROTQ9S-0001`, paid), one real prediction run (photo + recorded voice) through the Playground. Result: a real 2-second output video, `predict_time` 16.8s, "Approximate cost" shown on Replicate's own page: **$0.10** - and the account's own real Billing page showed "Usage this month: $0.10" **at the same moment**, matching exactly. Unlike fal.ai (where the real invoice never once matched what the code requested or fal.ai's own Request History confirmed receiving), **Replicate's quoted-at-generation-time cost is the real billed cost, with no discrepancy found**. The "$0.02 per second of output video" headline on the model's marketing page does not hold cleanly for a short 2s clip - $0.02 x 2s = $0.04, not $0.10 - because real cost tracks real compute time (16.8s) more than output length, which is an honestly-disclosed number (shown on the page, matching billing exactly), not a hidden one.

**Built `ReplicateVideoClient` (`src/integrations/replicate_video_client.py`)** - same interface shape as `FalVideoClient` (`generate_speech_to_video(*, prompt, image_data_uri, audio_data_uri, resolution="480p")`, `resolution` accepted only for interface compatibility and ignored, since this model has none), `RealReplicateVideoClient` (real httpx calls, submit → poll `urls.get` → download `output`, defensive against three real output shapes since Replicate's own docs never document this model's exact output format), `MockReplicateVideoClient` (same call-recording pattern as `MockFalVideoClient`).

**`PROVISIONAL_PRICE_PER_SECOND_USD = 0.05`** - the one real data point above ($0.10 / 2s), kept explicitly as a conservative, non-final placeholder (clearly labeled in the module docstring) until more real runs either confirm or refine it. This is deliberately the HIGHEST real rate observed so far, not a hopeful lower one - the same safety posture `REAL_BILLED_PRICE_PER_SECOND_USD` takes for fal.ai.

**A real structural gap this caused, found and fixed before it could repeat the fal.ai mistake:** `DialoguePipeline` and `BotConversation` both used to import and hard-code fal.ai's own `REAL_BILLED_PRICE_PER_SECOND_USD` directly for all real-money tracking (the pre-Approve estimate and the `MAX_EPISODE_COST_USD` ceiling). Simply swapping in a Replicate client without this fix would have silently kept computing real cost at fal.ai's rate - the EXACT "wrong rate used without anyone noticing" defect this project spent the whole evening fixing for fal.ai itself, just reintroduced from a different direction. Both classes now take an explicit `price_per_second_usd` parameter (still defaulting to fal.ai's real rate, so every existing call site keeps working unchanged); `src/app.py` is the one place that decides which real provider is wired in and passes the matching real rate to both.

**Provider selection (`src/config.py`, `src/app.py`):** `VIDEO_PROVIDER` env var explicitly picks `"fal"` or `"replicate"`; left unset, Replicate wins when configured (the provider the owner asked to move to), falling back to fal.ai so an existing deployment with only `FAL_KEY` set keeps working unchanged. Character portrait generation (`fal-ai/flux/dev`) is unaffected either way - a separate real endpoint, still on fal.ai, not part of this migration.

**He also asked how to give Claude access to his Replicate account** - same answer as fal.ai: never share a login. He generated a real API token through Replicate's own UI and briefly pasted its real value directly into this chat before being told to revoke it immediately - the token was never used, and he was walked through the correct, safe path instead: a GitHub repository secret (`REPLICATE_API_TOKEN`), wired into `.env` on every deploy via `appleboy/ssh-action`'s own documented `envs:` mechanism (verified against the action's own README, 2026-10-21) - upserted, never printed, same convention as every other real key in `.github/workflows/deploy.yml`.

Covered by `tests/test_replicate_video_client.py` (12 tests: submit/poll/download, all three real output shapes seen in practice, the resolution kwarg is accepted but never sent, every real failure path mirroring the existing fal.ai client's own coverage), `tests/test_app.py` (5 tests: auto-select Replicate-only, auto-select fal-only, Replicate preferred when both configured, explicit `VIDEO_PROVIDER` override wins, no real pipeline when neither is configured), and new tests in `tests/test_dialogue_pipeline.py`/`tests/test_bot_conversation.py` proving a custom `price_per_second_usd` is actually honored (0.05, deliberately distinct from both fal.ai constants so a pass can't be coincidental). Mutation-tested: reverted the price-parameter wiring in both classes back to the hard-coded fal.ai constant, confirmed both new cost tests fail for the real reason (obtained fal.ai's rate, expected the custom one); separately reverted `app.py`'s provider-preference order, confirmed the "prefers Replicate" test fails for the real reason (got `RealFalVideoClient` instead). Both reverted and restored; full suite green (168/168).

## fal.ai removed from the active path entirely - video and images both (2026-10-21)

Minutes after Replicate shipped for video, the owner asked who builds the character portraits now. Told the honest architecture at that point - portraits still went through fal.ai (`fal-ai/flux/dev`), a separate real endpoint from video, untouched by the migration above. His reply, verbatim and unambiguous: **"لا تضع السراق، بل احذفهم من الكود على الاقل الان"** (don't put the thieves in, remove them from the code, at least now) - fal.ai is not to be called for anything any more, not just the part that was proven to overbill.

**Built `ReplicateImageClient` (`src/integrations/replicate_image_client.py`)**, mirroring `ReplicateVideoClient`'s own shape: real calls to `black-forest-labs/flux-dev` on the same platform-wide contract already verified for the video model (`https://replicate.com/docs/reference/http`), defensive `_extract_image_url()` handling every output shape plausible for a Replicate model (string/list/dict), same reasoning as the video client's own `_extract_video_url()`.

**Said plainly, not hidden**: this model's own exact input schema and its real live price could **not** be independently confirmed the way the video model's were - `WebFetch` against both `replicate.com/black-forest-labs/flux-dev` and its `/api` page returned no pricing and no input schema (client-rendered content), and the public model-metadata API (`GET /v1/models/...`) requires an authenticated token this session does not hold. `prompt` is the one field name confirmed from Replicate's own public blog posts about this model. `PRICE_PER_IMAGE_USD_UNVERIFIED = 0.025` is carried over from fal.ai's own figure as a labeled placeholder only, not a confirmed Replicate price - unlike the video rate (a real, owner-run, owner-confirmed test), no equivalent real test has been run for images yet. Flagged in the module's own docstring so this gap is never silently forgotten.

**`src/app.py` now ties the image provider to the SAME `chosen_video_provider` decision as the video client**, not to a separate `fal_configured` check - the only way to guarantee fal.ai is never constructed, let alone called, once Replicate is chosen, even though the real droplet still has a valid `FAL_KEY` sitting in `.env` (kept only so `VIDEO_PROVIDER=fal` remains a real, working fallback if ever needed - but never reached by default any more).

Covered by `tests/test_replicate_image_client.py` (8 tests, same pattern as the video client's own suite) and two new tests in `tests/test_app.py`: `test_build_app_never_constructs_a_fal_client_when_replicate_is_chosen` (asserts `app.fal_video` and `app.dialogue_pipeline._image_client` are BOTH Replicate classes, and explicitly asserts they are NOT the fal.ai classes, even with `FAL_KEY` also set - the exact real condition on the real droplet) and `test_build_app_uses_fal_for_images_too_when_fal_is_explicitly_chosen`. Mutation-tested: reverted the image-client wiring back to always constructing `RealFalImageClient`, confirmed `test_build_app_never_constructs_a_fal_client_when_replicate_is_chosen` fails for the real reason (`isinstance(..., RealReplicateImageClient)` was `False`, got `RealFalImageClient` instead), restored, confirmed the full suite green (178/178).

## Manual photo upload: the owner's own photos, zero generation cost (2026-10-21)

The owner sent a real, finished grid of six character portraits (his own real screenplay's cast - a mother, a young man, his father, a divorced woman, her father, and a taxi driver named موعز) and asked directly: "الصور ضعها انت الان و هكذا لا ندفع شيء" (you put the photos in yourself, this way we don't pay anything). He does not want to spend any real money generating photos he already has.

**`DialoguePipeline.save_character_photo(title, character_name, image_bytes)`** (new, `dialogue_pipeline.py`) writes directly to the EXACT real cache path `_resolve_character_photo()` already checks first and trusts unconditionally (`{output_dir}/characters/{slug(title)}_{slug(character_name)}.jpg`) - the same cache-hit short-circuit a previously-generated photo already relies on. A photo saved this way costs nothing and triggers no real generation call at all, for either provider, the next time this exact screenplay is produced.

**New `/photo` message handler (`runner.py`)**: the owner sends a real photo to the bot with the character's exact name (as it appears in the current screenplay) as the caption. `BotConversation.get_current_screenplay(chat_id)` (new, read-only accessor) validates the name against the real current draft before anything is saved - a typo or wrong name gets back the real list of names in that screenplay, not a silent miss. The real photo bytes are downloaded via python-telegram-bot's own async file API and written through the executor (same pattern as every other real, if fast, I/O call this file makes).

This is architecturally correct, not a one-off hack: it reuses the SAME cache path the automatic pipeline already checks first, so there is exactly one mechanism deciding "do I need to pay to generate this photo" - not two competing ones. Works identically whichever video/image provider is currently chosen (Replicate or fal.ai), since the cache check in `_resolve_character_photo()` happens before either provider is ever touched.

Covered by `tests/test_dialogue_pipeline.py` (`test_save_character_photo_is_picked_up_as_a_cache_hit_with_zero_real_cost` - produces a real screenplay for the exact character after saving its photo, asserts the image client is never called, the real cost is $0, and the real bytes sent to the video client for that character match exactly what was saved; `test_save_character_photo_overwrites_an_existing_cached_photo`) and `tests/test_bot_conversation.py` (3 tests for `get_current_screenplay`: none before any screenplay, the real current draft once one exists, none again after `/cancel`). Mutation-tested: changed the real cache filename `save_character_photo()` writes to (`..._manual.jpg` instead of the exact path `_resolve_character_photo()` checks), confirmed the cache-hit test fails for the real reason (`assert 1 == 0` - a real generation call happened because the paths no longer matched), restored, confirmed the full suite green (183/183).

## Replicate's real low-credit throttle - every portrait failed in one real production run (2026-10-21)

The first real production run on Replicate failed outright - every single character's portrait, real error text relayed verbatim by the owner:

```
Replicate image submit failed (429): {"detail":"Request was throttled. Your rate
limit for creating predictions is reduced to 6 requests per minute with a burst
of 1 requests while you have less than $5.0 in credit. Your rate limit resets
in ~6s.","status":429,"retry_after":6}
```

This is Replicate's own, real, documented behavior - an account with under $5.00 in credit is throttled to 6 requests/minute with a burst of exactly 1. `DialoguePipeline`'s own pre-pass loop submits one portrait-generation request per speaking character back-to-back, with no pacing at all - exactly the pattern this real throttle exists to reject. Neither `RealReplicateImageClient` nor `RealReplicateVideoClient` had any retry logic for this (or for anything) - a single 429 killed that character's portrait outright, with the real, correct, no-fallback-face consequence already documented elsewhere: every line of that character failed and was reported, not silently replaced.

**Fixed, using the real information the API itself provides**: both clients' submit call now goes through `_post_with_retry()` - on a real `429`, it reads `retry_after` (seconds) straight out of the response body Replicate already sent and sleeps exactly that long before retrying, up to `MAX_SUBMIT_RETRIES = 5` attempts, then gives up for real (an account that is genuinely out of credit must still fail cleanly, not retry forever). This is the same model this project already applies everywhere else: never guess a backoff duration when the real server already states one.

Both `replicate_image_client.py` and `replicate_video_client.py` got the identical fix - the video endpoint shares the same account and the same real risk from the dialogue loop's own back-to-back line submissions.

Covered by 4 new tests (2 per client, identical shape): `test_generate_image_retries_a_real_429_and_honors_the_real_retry_after` (a 429 then a real success - asserts exactly one retry and that `time.sleep` was called with the real `retry_after` value from the response, not a guessed constant) and `test_generate_image_gives_up_after_max_retries_on_persistent_429s` (asserts exactly `MAX_SUBMIT_RETRIES` attempts before the real error propagates), mirrored for the video client. Mutation-tested: reverted the submit call back to a bare `httpx.post` with no retry, confirmed both new tests fail for the real reason (one retry became zero; 5 expected attempts became 1), restored, confirmed the full suite green (187/187).

## Voice dialect reversed again, Syrian → Tunisian (2026-10-21)

A real produced episode (10 real dialogue clips, $2.552 real cost, the same run whose 12 other lines hit the 429 throttle above) was the first real test of the 2026-10-19 Syrian-voice switch under real conditions. The owner's real verdict, listened to on his own phone: the dialect was NOT clear to him - "لهجة عربية مختلفة" (a different Arabic dialect [than what I understand]) - directly contradicting the 2026-10-19 reasoning ("السورية ... بيضاء ومفهومة جداً لجميع سكان شمال إفريقيا").

This surfaced alongside a separate, serious accusation worth recording honestly: the owner initially described the result as "كله راوي" (all narrator), reading the combination of a 429-truncated episode (10 of 22 lines produced, 12 missing entirely) as the forbidden narrator pattern returning. The real evidence he then sent (a screenshot of the bot's own "تم. 10 مقطع حوار، التكلفة الحقيقية $2.552" / "فشلت 12 سطرًا" message) showed no "راوي" character anywhere - real per-character dialogue clips and real per-line failures, exactly the documented no-fallback-face behavior, not narration. Shown the real evidence rather than disputed from memory; the owner did not pursue the narrator claim further once the 429 explanation and its fix were laid out. Recorded here, not quietly dropped, because "no narrator" is the single most load-bearing guarantee in this codebase and a report of it recurring deserves exactly this level of scrutiny every time, confirmed or not.

**Reversed**: `character_photos.py`'s `_VOICE_BY_GENDER` and `models.py`'s `Character.voice` default both switched back to `ar-TN-HediNeural` (male) / `ar-TN-ReemNeural` (female) - the exact same real, live-verified Tunisian voices already on file (docs/SOURCES.md "Azure TTS voiceover"), not a new unverified choice. Both the Syrian and Tunisian voice sets have now been real-tested in production and found wanting / acceptable respectively by the one person whose ear is the actual spec here.

New `tests/test_character_photos.py` (3 tests) locks in the exact real voice strings, case/whitespace handling, and the male fallback for an unrecognized gender - the existing `test_write_dialogue_screenplay_assigns_different_voices_by_gender` only ever asserted the two voices differ from each other, which would have passed under either dialect and caught nothing about which one. Mutation-tested: reverted the dict back to the Syrian voices, confirmed all 3 new tests fail for the real reason (`ar-SY-LaithNeural` obtained where `ar-TN-HediNeural` was expected), restored, confirmed the full suite green (190/190).

## Owner's instruction, "قم بالتنظيف" (clean up) - the portrait price had the identical defect already fixed for video (2026-10-21)

Responding to the pace of changes this session, the owner asked for a cleanup pass before anything else, and separately showed a real Replicate billing screenshot ($8.42 credit remaining, well above the $5 throttle threshold) disputing that the earlier 429 fix was addressing a real cause - a fair challenge given at the exact moment of that real failure the balance genuinely was under $5 (confirmed by the 429 body itself, which only fires under that condition), but worth a real review regardless rather than asserting that from memory.

The review found a real, concrete instance of exactly the class of bug the owner is worried about: `DialoguePipeline._resolve_character_photo()` and `BotConversation._format_screenplay()` both still imported and hard-coded fal.ai's own `PRICE_PER_MEGAPIXEL_USD` for the portrait cost component - the *exact* defect already found and fixed for video seconds (`price_per_second_usd`) two commits earlier, just never caught for the image side. It went unnoticed specifically because `PRICE_PER_MEGAPIXEL_USD` (fal.ai, $0.025) and `PRICE_PER_IMAGE_USD_UNVERIFIED` (Replicate's placeholder, also $0.025) happen to be numerically equal right now - the moment either price is independently confirmed and they diverge, every portrait-cost number shown to the owner (the pre-Approve estimate and the real `MAX_EPISODE_COST_USD` ceiling) would have silently reverted to fal.ai's rate on a Replicate-wired build.

Also found and fixed in the same pass: `_resolve_character_photo()`'s own "no image client configured" failure message still named `FAL_KEY` specifically ("لا يوجد اتصال حقيقي بمولّد الصور (FAL_KEY غير مُهيَّأ)") - wrong and misleading on a Replicate-only deployment where the real missing credential would be `REPLICATE_API_TOKEN`. Now names both.

**Fixed**: both `DialoguePipeline` and `BotConversation` now take an explicit `price_per_image_usd` parameter (default unchanged, so every existing call site keeps working), exactly mirroring `price_per_second_usd`'s own fix. `src/app.py` passes the correct real constant for whichever provider is actually chosen - `PRICE_PER_IMAGE_USD_UNVERIFIED` for Replicate, `PRICE_PER_MEGAPIXEL_USD` for fal.ai.

**A real gap in the test itself, caught before it shipped**: the first version of the `test_app.py` coverage asserted `app.dialogue_pipeline._price_per_image_usd == PRICE_PER_IMAGE_USD_UNVERIFIED` - which passed even when deliberately mutated to wire in the WRONG (fal.ai) constant, because the two real constants are still numerically equal. Replaced with `test_build_app_wires_replicates_own_image_price_not_fals`, which monkeypatches `PRICE_PER_IMAGE_USD_UNVERIFIED` to `0.4242` (a value that cannot coincide with anything) before calling `build_app()` - this is the only way to prove the *wiring* is correct independent of the two constants' current accidental equality. Mutation-tested properly this time: confirmed the monkeypatch-based test fails for the real reason when the wiring is reverted to the wrong constant, restored, confirmed the full suite green (193/193).

Separately: re-confirmed the no-narrator guard (`_NARRATOR_NAME_PATTERN = re.compile(r"راوي|narrator", re.IGNORECASE)` in `dialogue_writer.py`) is intact and unchanged by any of tonight's work - grepped the full active codebase for "راوي"/"narrator" outside comments and found only this one real, active enforcement point. Recorded here because the owner raised the narrator claim seriously enough to warrant checking, not because anything was found wrong.

## fal.ai deleted from the codebase, not just bypassed (2026-10-21)

The owner's instruction up to this point ("الان اريد البوت بربليكات") had only made Replicate the *preferred* real provider, with fal.ai kept as a selectable fallback (`VIDEO_PROVIDER=fal`) and still the real client wired in for images by default. The owner rejected that as insufficient, explicitly and with rising urgency across three messages: "لا تضع السراق، بل احذفهم من الكود" (don't put the thieves [fal.ai] in, delete them from the code), then "بل احذف كل الذي طلبته منك و خذ وقتك، هذا ما طلبته منك بالظبط" (delete everything I asked, take your time, this is exactly what I asked), then "الان احذف" (now delete). Real reason already on file: a real fal.ai invoice proved it billed every speech-to-video call at the 720p rate regardless of what the code requested (see above) - the owner no longer wants that code path reachable under any configuration, not even as a fallback nobody enables.

**Deleted outright, not merely made unreachable**:
- `src/integrations/fal_video_client.py` and `src/integrations/image_client.py` - fal.ai's own real client code - removed from the repository entirely.
- `tests/test_fal_video_client.py` and `tests/test_image_client.py` - removed entirely.
- `Settings.fal_key`, `Settings.fal_configured`, and `Settings.video_provider` (`src/config.py`) - the explicit-override/auto-select logic between fal.ai and Replicate no longer exists, because there is only one real provider left to select.
- `src/app.py`'s `build_app()` - the `chosen_video_provider` branching (auto-select, explicit `VIDEO_PROVIDER=fal`/`=replicate`) is gone. Replicate is now the only real path; an unconfigured `REPLICATE_API_TOKEN` (or missing `AZURE_SPEECH_KEY`) means no real pipeline at all, same "no silent stand-in" posture already applied to character portraits - never a silent fall-back to fal.ai, because fal.ai no longer exists to fall back to.
- `src/workflows/dialogue_pipeline.py` and `src/bot/conversation.py` - default `price_per_second_usd`/`price_per_image_usd` now default to Replicate's own constants (`PROVISIONAL_PRICE_PER_SECOND_USD`, `PRICE_PER_IMAGE_USD_UNVERIFIED`) instead of fal.ai's.
- `src/main.py` - the "no real pipeline" error message used to name `FAL_KEY` specifically; now names `REPLICATE_API_TOKEN`, the credential that is actually checked.
- `.env.example`, `README.md`, `.github/workflows/deploy.yml` (the real key-presence diagnostic printed on every deploy) - all updated to drop `FAL_KEY`/`VIDEO_PROVIDER` as options.

**Left alone, deliberately**: comments in `src/models.py`, `src/agents/dialogue_writer.py`, and elsewhere that narrate *why* a past fix looked the way it did (e.g. the 2026-10-19 Syrian-voice switch, the original 720p/480p billing finding) still name fal.ai - that is real history, the same way this file keeps every past incident on record rather than rewriting it. The two docstring mentions that described *current* behavior (`models.py`'s `Character` docstring, `dialogue_writer.py`'s module docstring) were updated to point at `replicate_video_client.py` instead, since those were describing the live code path, not history.

Test suite rewritten to match: `tests/test_dialogue_pipeline.py`, `tests/test_bot_conversation.py`, and `tests/test_app.py` now use `MockReplicateVideoClient`/`MockReplicateImageClient`/`ReplicateImageError` throughout instead of their fal.ai equivalents, and `test_app.py`'s provider-selection tests (auto-select fal vs Replicate, explicit override, "never constructs a fal client") were removed outright - there is no longer a choice to test, only "is Replicate configured or not" (`test_build_app_has_no_real_pipeline_when_replicate_is_not_configured`). Full suite green: 175/175 (down from 193 - the fal.ai-specific client tests and the now-impossible provider-choice tests were the difference, not a loss of real coverage).

**Mutation-tested the real wiring, not just the deletion**: temporarily hard-coded `price_per_image_usd=0.025` in `app.py`'s real-Replicate branch (reverting the fix this change still depends on), confirmed `test_build_app_wires_replicates_own_image_price_not_a_stale_constant` fails for the real reason (`assert 0.025 == 0.4242`), restored the real wiring, confirmed the full suite green again (175/175). A grep for every fal.ai symbol (`fal_video_client`, `FalVideoClient`, `RealFalVideoClient`, `MockFalVideoClient`, `REAL_BILLED_PRICE_PER_SECOND_USD`, `FalVideoError`, `image_client` module imports, `RealFalImageClient`, `MockImageClient`, `ImageGenerationError`, `PRICE_PER_MEGAPIXEL_USD`, `FAL_KEY`, `VIDEO_PROVIDER`) across `src/` and `tests/` after the change returns zero functional matches - only the deliberately-kept historical comments above.

Also fixed in the same pass, found by that grep, not asked for separately: `src/main.py`'s own "no real pipeline" log line still said `FAL_KEY and/or AZURE_SPEECH_KEY not set` - a real, functional bug (the wrong credential name in a message an owner with no computer or logs has to act on), now says `REPLICATE_API_TOKEN and/or AZURE_SPEECH_KEY not set`.

## The 429 throttle recurred - this time above the owner's own $5 threshold, and the real cause stayed unresolved (2026-10-21)

The owner sent a real screenshot: a produced episode where one character's portrait ("الطليقة") failed with the exact same real 429 already documented above ("reduced to 6 requests per minute with a burst of 1 requests while you have less than $5.0 in credit"), despite the 429-retry fix from earlier the same night already being live and unchanged. The same message also showed a real, successfully stitched 0:47 video with the "نشر على YouTube"/"تجاهل" buttons - so a real film WAS produced, contradicting the owner's own read of it as "لا يوجد فيلم اصلا" (no film at all) literally, though at least one character's lines were genuinely missing.

The owner then sent a second real screenshot - Replicate's own `account/billing` page, showing **$8.42 credit remaining right now**, a **$10.00 credit purchase succeeded 10/21**, and $1.58 used this month - directly challenging the earlier "low credit" explanation for the original 429 fix, and rejecting it outright ("المشكلة ليست مشكلة رصيد").

**First research pass** (web search against Replicate's own docs, https://replicate.com/docs/topics/predictions/rate-limits): confirmed the exact numbers in the error (6/minute, burst of 1) match a documented tier, but tied to accounts that have "been granted credit and don't have a payment method on file" - a different, narrower condition than raw balance. Found independent corroboration: a different real user hit the identical error text at a real $9.56 balance (Inflearn community thread, https://www.inflearn.com/en/community/questions/1832677) - confirming this is not unique to this account, and that balance alone does not explain it.

**The owner correctly rejected the "no payment method" theory too** ("كيف احفظ بطاقتي عندهم و الرصيد موجود... هذا ليس شرط منهم"). **A second, deeper research pass confirmed the owner was right**: Replicate's own billing docs (https://replicate.com/docs/topics/billing/prepaid-credit) state, as a hard sequential fact, that a payment method must already be on file *before* any credit purchase can be processed at all - so the real, successful $10.00 purchase is itself proof a payment method existed at that time. The "no payment method on file" explanation does not hold up against Replicate's own documented purchase mechanics.

**Honest conclusion, stated plainly rather than guessed past**: no public Replicate source - official docs, changelog, status page, or GitHub - explains why this exact message fires for a real account above $5 credit with a payment method on file. Two genuine ambiguities remain in Replicate's own docs (the undefined term "granted credit," and whether a payment method must be marked "default" to count) that only Replicate's own support (https://replicate.com/support) could resolve definitively - not something guessable from outside.

**Fixed anyway, without needing to resolve the mystery**: `src/integrations/replicate_rate_limiter.py` (new) - a `SubmitPacer` shared between `RealReplicateImageClient` and `RealReplicateVideoClient` (they bill against the same real Replicate account, therefore the same real rate-limit bucket) that paces every real submission to never exceed Replicate's own documented worst-case restrictive tier (60s / 6 requests = `MIN_SECONDS_BETWEEN_SUBMITS = 10.0`), calling `time.sleep()` for the real remaining gap before a new submission goes out - proactive, not just reactive to a 429 after the fact. This is a new, separate layer from `_post_with_retry()`'s existing 429-retry logic, which still honors whatever `retry_after` Replicate's own response states for retries within one submission.

5 new tests (`tests/test_replicate_rate_limiter.py`): the pacer's own timing math in isolation (first call never waits, a call inside the window sleeps the real remaining gap, a call after the window never waits, `reset()` genuinely clears state - built so the test itself fails if `reset()` is a no-op), plus a check that `MIN_SECONDS_BETWEEN_SUBMITS` is derived from Replicate's own stated "6 requests per minute," not a separately guessed number. Plus one integration test per client (`test_generate_image_paces_itself_through_the_shared_submit_pacer`, `test_generate_speech_to_video_paces_itself_through_the_shared_submit_pacer`) proving each real client actually calls the shared pacer before submitting, not a copy that could silently drift. `tests/conftest.py` gained an autouse fixture resetting the shared pacer's state before and after every test, since it is real, shared, module-level state that would otherwise leak real timing between unrelated tests.

Mutation-tested twice: (1) removed the `shared_submit_pacer.wait_for_slot()` call from both clients - both new integration tests failed for the real reason (`assert [] == [1]`), restored, confirmed identical to the original files. (2) disabled the elapsed-time check inside `SubmitPacer.wait_for_slot()` itself (`if self._last_submit_monotonic is not None` → `if False`) - the pacing-math test failed for the real reason (`assert [] == [7.0]`), restored. Full suite green both times, 182/182 at the end.

## A real screenplay, then a raw, uncaught error - not Replicate, not fal.ai, a different real gap (2026-10-22)

The owner sent a real screenshot: a genuine, complete screenplay (a long dramatic scene) delivered successfully at 01:51, then six minutes later, at 01:57, a raw error reached him verbatim through Telegram: `❌ حدث خطأ حقيقي غير متوقع: INTERNAL. {'error': {'code': 500, 'message': 'Internal error encountered.', 'status': 'INTERNAL'}}`.

Traced to its real source: `src/bot/runner.py`'s generic top-level error handler (`context.error`) - meaning the real exception had escaped every other handler untouched. The error shape (`{'error': {'code': 500, 'status': 'INTERNAL'}}`) matches `google.genai.errors.ServerError`'s own string form exactly - a real Gemini 500, reached after Groq's own real call had already failed and fallen back to Gemini (`RealGroqReasoningClient`'s documented fallback), whose own 3 built-in retries (`RealGeminiReasoningClient._MAX_RETRIES`) were then also genuinely exhausted. A real, rare double-provider outage, not a bug in either client - but the bot had no graceful way to tell the owner that.

**Root cause, confirmed by reading the real code, not guessed**: `DialogueWriterAgent.route_message()` and `_write()` (`src/agents/dialogue_writer.py`) called `self._reasoning_client.complete(...)` directly, with no try/except at all. Only JSON-parsing/validation failures were ever wrapped into `DialogueWriterError` - a real network/provider exception (any `Exception` the reasoning client itself raises) propagated straight past `conversation.py`'s own `except DialogueWriterError` guards (which cannot catch an exception of a different type) all the way to the generic handler.

**A second, separate real gap found in the same review**: `BotConversation.handle_new()` and `handle_full()` (`src/bot/conversation.py`) never caught `DialogueWriterError` around their own calls to `write_dialogue_screenplay()`/`write_complete_dialogue_screenplay()` at all - unlike the equivalent free-conversation `build_screenplay` branch inside `handle_message()`, which already does. So even a genuine, already-documented `DialogueWriterError` (e.g. a malformed JSON reply) would crash `/new`/`/full` the same way, while the free-conversation path handled it cleanly. This asymmetry is also why `src/bot/runner.py`'s own `_route_after_debounce` (the free-text path) already has a defensive `try/except Exception` wrapping `conversation.handle_message()` - its own 2026-10-21 comment already names this exact failure class ("RealGroqReasoningClient/RealGeminiReasoningClient both raise raw exceptions, never DialogueWriterError, on a real API/network failure") - but `_new`/`_full` in the same file never got the matching protection.

**Fixed at the real source, not just patched at the symptom**: `DialogueWriterAgent` gained one new method, `_complete()`, that every real call site (`route_message`, and both calls inside `_write`) now goes through instead of calling `self._reasoning_client.complete()` directly - it catches any non-`DialogueWriterError` exception and re-raises it as a clean `DialogueWriterError` naming no real money was spent. `BotConversation.handle_new()`/`handle_full()` now wrap their screenplay-writing calls in `except DialogueWriterError`, mirroring the free-conversation branch exactly. `src/bot/runner.py`'s `_new`/`_full` handlers also gained the same defense-in-depth `try/except Exception` the free-text path already has, for any other unexpected failure outside this specific class.

6 new tests: 3 in `tests/test_dialogue_writer.py` (`_AlwaysFailingReasoningClient`, a fake that raises a plain `RuntimeError` the same shape as the real incident) proving `route_message()`, `write_dialogue_screenplay()`, and `write_complete_dialogue_screenplay()` each convert it into a real `DialogueWriterError`; 2 in `tests/test_bot_conversation.py` proving `handle_new()`/`handle_full()` turn that into a clean `BotReply` (no buttons, no crash) instead of propagating. Mutation-tested: (1) reverted `_complete()` to a direct `self._reasoning_client.complete()` call at all 3 sites - all 3 new dialogue_writer tests failed for the real reason (the raw `RuntimeError` reached the assertion instead of `DialogueWriterError`), restored, confirmed identical to the original file. (2) removed the new `try/except DialogueWriterError` from `handle_new`/`handle_full` - both new conversation tests failed for the real reason (the `DialogueWriterError` propagated uncaught past the test), restored. Full suite green both times, 187/187 at the end.

Also real evidence this was never a Replicate or fal.ai issue, despite arriving in the same conversation as both: the error text, the 01:51→01:57 timing (6 minutes, consistent with a screenplay-writing follow-up message, not production - which only starts after pressing ✅ موافقة), and the code path traced all point to the reasoning client (Groq/Gemini), never touched by either the fal.ai deletion or the Replicate rate-limit pacing work earlier the same session.

## A duration dispute that turned out to be a Telegram display artifact, then a real narrator complaint that wasn't (2026-10-22)

The owner, after the previous fix shipped, sent a real screenshot of a produced episode: `✅ تم. 22 مقطع حوار، التكلفة الحقيقية $6.075`, published successfully to YouTube, with a video preview showing a real duration of `0:13` next to it - and asked why the first video ever produced worked but nothing since had. A real, reasonable question given that number: $6.075 at `PROVISIONAL_PRICE_PER_SECOND_USD` implies roughly two minutes of real billed audio, not 13 seconds.

Checked with real, independent evidence rather than disputed or accepted at face value: fetched the actual published video (`https://youtu.be/AfdkAOVAtIE`) via `yt-dlp` (installed for this one check) - **duration_seconds: 179** (2:59), not 13. The owner's own phone later confirmed the same thing directly, playing the real file: `02:24 / 02:58`. Telegram's own preview thumbnail had shown an inaccurate, not-yet-loaded duration label (a known client behavior on a slow connection, visible in the owner's own status bar) - the real delivered file was never short or broken. Reported back with both pieces of real evidence, not just a reassurance.

**That resolved, but surfaced the real complaint underneath it**: "الراوي اكثر من ثلاثة دقائق، مشكلتي ليست الدقائق بل الراوي" (the narrator issue is beyond three minutes, my problem isn't the duration, it's the narrator) - then, concretely: "كله الفيديو كله صورة شخص يسرد قصة" (the entire video is one person's photo narrating a story), confirmed directly when asked whether the video had multiple characters' voices interacting or one voice the whole time: "صوت واحد فقط طوال الفيديو" (one voice only, the entire video).

**Root cause, found by reading the real code, not guessed**: `_NARRATOR_NAME_PATTERN` (the existing code-level guard) only ever checks whether a cast member's **name** contains "راوي"/"narrator" - it has no way to catch a real, legitimately-named character (e.g. "الشاب") being used **as** a narrator in practice: carrying nearly the entire screenplay alone, in third-person self-description, while the rest of a real multi-person cast gets little or nothing to say. That is functionally a narrator wearing a human name, and the prompts' own "no narrator" instructions (`_LINES_SYSTEM_PROMPT`, `_COMPLETE_LINES_SYSTEM_PROMPT`) are only ever instructions the model can still fail to follow - exactly the reasoning `_NARRATOR_NAME_PATTERN` itself was already built on, just never extended to cover this specific failure shape.

A quick, real test against the exact text the owner had sent earlier in this conversation (`count_quoted_moments()` run directly against it) confirmed the mechanism is plausible: of a 562-character dramatic passage, only 71 characters (12.6%) were inside real quote marks - the rest (third-person narrative description, e.g. "وهو منذ ذلك الحين.. جسد بلا روح، يفكر فيها وحدها...") is exactly the kind of content that should never become a spoken `line`, but nothing in `_COMPLETE_LINES_SYSTEM_PROMPT` deterministically stops a single quoted span from itself reading as narration, or stops the model from attributing nearly every real line to one character regardless.

**Fixed with a new, deterministic, code-level gate** (not another prompt tweak alone): `DialogueWriterAgent._write()` - shared by both `/new` and `/full` - now counts lines per character after parsing the model's real reply, and raises `DialogueWriterError` if any one character holds more than `_MAX_SINGLE_CHARACTER_LINE_SHARE = 0.7` (70%) of the total lines in a cast of 2 or more characters, before any real cost is spent. 0.7 is a real engineering choice: strict enough to reject the real reported case (one voice, the entire video), loose enough that one naturally talkative character among others who genuinely participate is not rejected for a plausible, legitimate imbalance. A single-character cast is exempt - nothing to balance against, and not the pattern this guard targets.

**Separately, found and fixed in the same review**: the owner's own screenshots of the live bot showed it repeating a near-identical boilerplate reply - "اعتذر... قد يكون هناك راوٍ... اضغط ✅ أسفل السيناريو للبدء في الإنتاج" - to five real, completely different messages in a row, including direct questions ("اي زر؟"، "هل هذا تعمد منك ام من الكودات؟") and open hostility ("لماذا لا تجيب يا غبي؟") - never once answering what was actually asked. Traced to `_ROUTER_SYSTEM_PROMPT`'s own "discuss" rule, which only ever told the model what to do when a message "sounds like approval" - nothing constrained the far more common case where it does not, so the model defaulted to the approval-reminder pattern regardless, and separately kept bringing up "راوٍ" unprompted, almost certainly triggered by seeing a caption line's own "(نص على الشاشة، غير منطوق)" marker in the draft shown to it on every call and conflating "not spoken" with "narration". Tightened: "discuss" replies must directly and specifically answer what the owner actually asked unless the message itself genuinely sounds like approval, and narrator must never be mentioned unprompted - a caption marker is explicitly named as not a narrator.

8 new tests total: 2 in `tests/test_dialogue_writer.py` asserting the updated router prompt text contains both new constraints (the project's own established style for prompt-engineering fixes - proves the instruction reaches the model, not that the model obeys it); 4 more covering the dominance guard (`write_dialogue_screenplay` and `write_complete_dialogue_screenplay` both reject a 3-of-4-lines/75% case naming the dominant character, allow a 2-of-3/67% case under the ceiling, and skip the check entirely for a single-character cast). Mutation-tested the dominance guard: replaced the real `if dominant_share > _MAX_SINGLE_CHARACTER_LINE_SHARE` check with `if False` - both reject-tests failed for the real reason (`DID NOT RAISE DialogueWriterError`), the allow/skip tests stayed green (proving the mutation didn't just break everything), restored, confirmed identical to the original file. Full suite green: 193/193.

## "الفرق بينهم في المشهد" - the real difference was the portrait's own background, not the dialogue (2026-10-22)

Immediately after the dominance-guard fix above, the owner sent a real, freshly-produced 2-line clip (balanced 1-1 between two characters, well under the new 70% ceiling) side by side with the project's first real video ever made (two people together in a real kitchen, a bowl of rice between them) and named the real remaining difference precisely: "الفرق بينهم في المشهد" (the difference between them is in the SCENE).

He had already, separately, rejected two wrong directions first: a proposal to add silent establishing/group shots between dialogue clips ("هل انت متاكد ام كذبة جديدة" territory - he wanted a fully spoken film, "فيلم ناطق فيه حوار بين الممثلين بالصوت والصوره", not another silent-film workaround), and a proposal to research true multi-person-in-one-frame lip-sync (real research done - `zsxkib/multitalk` on Replicate is real but a third-party community model capped at 2 people, unverified in production, ~$0.59/run flat; `alibaba/wan-3`, a real official text-to-video model, confirmed at $0.05/s@480p/$0.10/s@720p/$0.20/s@1080p, could generate silent group/establishing shots - but he explicitly didn't want that).

**Root cause, found by reading `PORTRAIT_STYLE_SUFFIX` itself**: `dialogue_pipeline.py`'s real reference-photo prompt has, since this constant was first written, always appended `"soft studio lighting, plain neutral background"` - every character's own real portrait (`black-forest-labs/flux-dev`) is deliberately generated as an isolated studio headshot against empty space, with zero scene/environment context, regardless of how the story's own dialogue is distributed between characters. No matter how real the lip-sync or how balanced the lines, a floating head in front of nothing can never read as a real film scene - this was never about narration or line distribution at all; it was a visual-generation gap the whole evening's narrator-focused investigation had never looked at.

**Fixed with no new model and no new real-money risk**: `PORTRAIT_STYLE_SUFFIX` rewritten to place the character "within a real, lived-in location that fits the story... never a plain studio backdrop or neutral background". `_resolve_character_photo()` now takes the real screenplay's own `premise` and includes the first 300 real characters of it in the image prompt ("in a scene from this real story: ...") so the image model has an actual setting to work from - capped, not the full premise verbatim, since this only needs to tell the model WHERE the character really is, not retell the whole story (and a long `/full` premise would otherwise bloat every single real image prompt for no benefit).

2 new tests in `tests/test_dialogue_pipeline.py`: the existing portrait test updated to assert the real new prompt shape; a new test proving the real premise text reaches the prompt and is genuinely capped at 300 characters, not sent in full. Mutation-tested: reverted the prompt construction back to the old studio-headshot-only version - both tests failed for the real reason (the new scene-context text was simply absent), restored, confirmed identical to the original file. Full suite green: 194/194.

Explicitly not solved by this, named honestly rather than overclaimed: two people do not yet appear TOGETHER in the same single frame - each character's clip is still generated and lip-synced separately, one person per shot. That would need a genuinely different, unproven model (`multitalk`, capped at 2 people, community-run) - real, available, but not something to build production on top of without a real paid test first, which is the owner's call to make, not something to decide here.
