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
