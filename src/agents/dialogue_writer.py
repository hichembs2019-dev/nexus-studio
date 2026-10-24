"""Dialogue Writer Agent: turns a one-line idea into a real character
screenplay - actual people speaking actual lines - the owner's real ask
("مسلسلات وأفلام ... لربح المال", series and films, not a narrated
documentary). Replaces WriterAgent's prose-beat output for any episode
meant to go through DialoguePipeline (real lip-synced speech, see
replicate_video_client.py): that pipeline needs a speaker attached to every
line it generates, not a block of narrative text a human would have to
re-parse for "who says what".

Same human gate as WriterAgent: the owner reads/edits
DialogueScreenplay.lines directly before DirectorAgent-equivalent work or
any real generation spends anything. Two LLM calls, not one, because
asking for both the cast list AND every line in a single structured
call regularly produced an inconsistent character roster (a name spelled
two ways is two different reference photos) - cast first, then lines
constrained to that exact roster.

Owner's instruction, 2026-10-19 ("الصور طبعا افتراضية... تُسند تلقائيًا
حسب وصف كل شخصية"): the cast step infers each character's gender and
voice is assigned automatically (see character_photos.py).

Owner's standing instruction, 2026-10-21 ("احذف الستة صور و قلت التوليد
حسب القصة" - delete the six photos, I told you generation must be
per-story): the six generic placeholder photos this agent used to assign
every character at screenplay-write time (`resolve_placeholder_photo`,
keyed by gender/age_group) are deleted outright, not merely stopped from
being used - see character_photos.py and dialogue_pipeline.py's own
docstrings. No photo is ever assigned here any more: `Character` carries
only `appearance` (the real per-story description DialoguePipeline
generates a real photo from during production). `age_group` is removed
from the cast step entirely - it existed only to pick a placeholder
photo size/style, and has no other consumer anywhere in this codebase.

Real finding, 2026-10-19 ("القصة يجب ان تكون كما طلبتها بالظبط بدون
تحويل" - the story must be exactly as I asked for it, no alteration):
the first real multi-line premise (a real, painful personal story,
already full of directly quoted dialogue in Arabic «guillemets») came
back with invented lines contradicting the premise outright - one
character answered immediately where the premise explicitly says she
stayed silent and answered neither yes nor no, and another was given a
line nobody said in the premise. `_LINES_SYSTEM_PROMPT` previously only
asked for dialogue that "flows" and is "plausible", which gave the model
room to invent freely even when the premise already supplied the real
words. It now explicitly requires reusing a premise's own quoted
dialogue verbatim where one exists, and forbids inventing a line a
character is explicitly described as NOT saying.

Second real finding, same story, same day ("اريد ممثلين يتكلمون لا راوي
يروي قصة" - I want actors talking, not a narrator telling a story): the
very bug this whole agent exists to fix (see the module docstring above)
came back through a side door. The premise is written as first-person
past-tense memoir prose ("كنت مطلقاً لزوجتي..."), so the cast step cast
the narrating "I" as a character literally named "الراوي" (The
Narrator), and the lines step then gave that character lines like
"فصرت أطلقها ولا أستمع لكلامها" (So I divorced her and stopped
listening to her) - a plain narrated summary of past events, not a line
anyone says to anyone in a scene. Both prompts now explicitly forbid a
"narrator" role: the real first-person narrator of a memoir-style
premise must be cast as an actual character (by name if the premise
gives one, otherwise by their real role - "الزوج"/"the husband" here,
never "الراوي"/"narrator") who appears in scenes and speaks real,
in-the-moment dialogue - prioritizing exactly the quoted lines the
fix above already requires reusing verbatim - never a line that
summarizes, recounts, or narrates events in past tense.

Third real finding, same story, same day ("انا كتبت قصة و هو يبترها" -
I wrote a story and it [the bot] cuts it): the SAME real 60-second
screenplay, checked line by line against the real story, DID cover the
full arc end to end (its last line was the story's real last line, the
mother's death) - nothing was truncated in transit this time. But the
15 lines it picked skipped the single most important quoted line in the
whole story: the mother's lie that is the real reason the marriage
ended («زوجتك شتمتك وسبتك في غيابك» - "your wife insulted and cursed
you behind your back"), and a couple of other structurally important
quoted moments, in favor of lines from elsewhere. With more real quoted
moments in the premise than there was room for, the model picked
somewhat arbitrarily rather than keeping the ones the plot actually
turns on. `_LINES_SYSTEM_PROMPT` now explicitly says so: when there are
more real quoted moments than {line_count} allows, keep the ones the
story's events actually depend on (what starts the conflict, each real
turning point, how it ends) over ones that are merely additional or
repeat a point already made.

Fourth real finding, same story, same day ("يجب ان ينفذ كل حرف و كل
كلمة كتبتها بدون حذف او نقصان ... أنا من أدفع المال و أنا حر ماذا
أريد" - every letter and word I wrote must be used, without deletion or
shortening - I'm the one paying, I'm free to choose what I want):
"prioritize the essential lines" (the third finding, above) is still a
selection - the owner explicitly does not want the model choosing AT
ALL when cost is not the owner's concern for that request.
`write_complete_dialogue_screenplay()` is a second, separate real mode:
line_count is never estimated from a target duration here - it is the
REAL number of quoted dialogue moments in the premise, counted by this
code with a plain regex over the premise text itself (`«guillemets»` or
"quotes"), not guessed or asked of the model. The model is then told
exactly that many lines are required, one per quoted moment, in order,
verbatim - no choosing, no dropping, no merging. The owner pays for
however many real lines that turns out to be (shown in the usual cost
estimate before approval) - that tradeoff is explicitly theirs to make,
not this code's to make for them.

Fifth real finding, 2026-10-21 ("لماذا تصر ان يوجد الراوي في الكودات ...
احذف الراوي للابد الا بطلب مني" - why do you insist the narrator exists
in the code, delete it permanently unless I ask): the fix above (the
second real finding) only ever told the model not to cast a narrator -
a prompt instruction the model can still ignore, and the real $6.838
incident's wrong video ("امراة تروي قصة" - a woman narrating a story)
is exactly what it looks like when it does. There was no code-level
check at all - a reply casting "الراوي"/"narrator" would have been
accepted and built into a real Character like any other name. `_parse_cast`
now rejects any such name outright with `DialogueWriterError`, before a
screenplay is ever built from it - a narrator can reach production only
if this check is deleted, never because the model happened to comply.

Owner's own definition of "فيلم سينمائي", 2026-10-23 ("فيه شخصيات
متعددة و مشاهد متعددة و حوار و موسيقى" - multiple characters AND
multiple scenes AND dialogue AND music): every screenplay this agent
ever wrote before this had exactly one implicit scene - the same
location, the same mood, for however many lines the premise produced,
because nothing here ever asked the model to think in scenes at all.
`DialogueScreenplay.scenes`/`DialogueLine.scene` (models.py) already
existed as inert, additive fields from the first step toward this - this
is the step that actually writes real values into them. Both line-writing
prompts now ask for a real "scene"/"location"/"mood" on every line, and
`_write()` groups the real parsed lines into real `DialogueSceneInfo`
entries from that - never asked of the model as a second, separate
array, because a scene's real cast (`characters_present`) is already
fully determined by which real characters actually speak inside it; a
second model-reported list of names could silently drift from the lines
it is supposed to describe, the same class of bug `_parse_cast`'s
duplicate-name check exists to prevent. `location`/`mood` ARE asked of
the model (real creative content, not derivable from structure), and
rejected here if either comes back empty - this is the first consumer
of those fields that will ever be downstream of a real decision (what
the owner reads before approving, eventually what a real music step
scores), so an empty placeholder here is not "not set yet" the way the
original additive field's own docstring allowed - it is a real gap in
output the model was explicitly asked not to leave.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.agents._text_parsing import parse_json_object, parse_json_object_array
from src.integrations.reasoning_client import AgentReasoningClient
from src.models import Character, DialogueLine, DialogueScreenplay, DialogueSceneInfo
from src.workflows.character_photos import resolve_placeholder_voice

_CAST_SYSTEM_PROMPT = """You are a casting director listing the speaking characters a short story needs.

Rules:
- Output ONLY a JSON array of objects, nothing else - no markdown fences, no commentary.
- Each object has exactly three keys:
  - "name": the character's name, as it should appear consistently everywhere (same spelling every time).
  - "gender": "male" or "female" - your best real inference from how the premise describes or refers to them.
  - "appearance": a short (one sentence, English) physical description for generating a REAL, photorealistic reference photo of THIS character, specific to this story - hair color/style, build, age, any distinguishing feature the premise mentions. If the premise gives no physical detail, invent a plausible, ordinary one consistent with their gender/role - never leave this generic or reuse the exact same wording for two different characters in the same cast.
- Only characters who actually SPEAK a line - no narrators, no silent background characters.
- NEVER create a character named/role "narrator", "راوي", "الراوي" or similar. If the premise is written as first-person memoir/narrative prose (e.g. "كنت..."/"I was..."), the person telling it is a REAL character who appears in the scenes and speaks real dialogue - cast them by their actual name if the premise gives one, otherwise by their real role in the story (e.g. "الزوج"/"the husband", "الأب"/"the father") - never as "narrator".
- 2 to 10 characters, names in the SAME language/script as the premise."""
# Real cap found too low, 2026-10-20: a real multi-scene /full premise
# (8 scenes, 26 real quoted lines) had 8 genuinely distinct speaking
# roles - protagonist, his own mother and father, his ex-wife, her
# father, his sister, a taxi driver, a neighbor. The old "2 to 6" cap
# would have forced the model to either merge two of them into one
# character (misattributing a real quoted line to the wrong speaker -
# exactly the correctness /full exists to guarantee) or drop a speaking
# role's lines entirely. Raised to 10 - a real short film can need more
# than six speaking parts; found by the first real story that actually
# needed more, not guessed in advance.

_LINES_SYSTEM_PROMPT = """You are a screenwriter turning a premise and a fixed cast into a real scripted scene - actual dialogue, not narration.

Rules:
- Output ONLY a JSON array of objects, nothing else - no markdown fences, no commentary.
- The array must have EXACTLY {line_count} elements.
- Each object has exactly these keys:
  - "character": one of these EXACT names (same spelling): {cast}
  - "line": what they actually SAY, in the SAME language as the premise, 1-2 short sentences a person would plausibly speak aloud - never a narrator's description of events. When the premise is in Arabic, every line you write (not a line copied verbatim from the premise) is correct, professional Modern Standard Arabic (العربية الفصحى) as spoken in a high-quality TV drama: grammatically correct, natural to say aloud, no dialect words mixed in, never a stiff word-for-word translation from English.
  - "action": a short, visual-only English stage direction for this moment (facial expression, gesture, what they're doing) - NEVER repeat or translate the spoken line itself here, the video model lip-syncs the real audio and would render duplicated text as more speech.
  - "scene": an integer, starting at 0 for the first scene and incrementing by 1 each real time the story moves to a genuinely different scene (a real change of location and/or a real time jump) - every line that stays in the same real scene as the line before it MUST repeat the SAME scene number. A short premise may be one real scene throughout (every line "scene": 0) - do not invent scene breaks that are not really there.
  - "location": a short (one sentence, English) description of WHERE this scene physically happens (e.g. "a warm family kitchen in the evening") - the SAME exact text on every line that shares this scene's number.
  - "mood": a short (a few words, English) emotional/atmospheric description of this scene (e.g. "tense and quietly heartbroken") - the SAME exact text on every line that shares this scene's number. This becomes the real cue a later music step scores this scene with - never a generic word like "neutral", describe what this specific scene actually feels like.
- The lines must flow as one continuous story in order, matching the premise, with characters actually responding to each other, scene by scene.
- This is an intense, emotional drama, not a report: every line carries a clear emotion in its own words (pleading, anger, grief, fear, tenderness, guilt, hope) and reacts to what the other character just said - a question answered, an accusation denied, a plea refused. Never a flat, calm statement of facts, and never two characters taking turns describing events. The "action" of each line names the real facial expression and gesture of that emotion.
- The premise is the real source of truth, not a loose inspiration: if it already contains a character's actual words (quoted directly, e.g. in "quotes" or «guillemets»), use that EXACT wording for their line - do not paraphrase or invent a different line for a moment the premise already gives you.
- Never give a character a line the premise explicitly says they did NOT say, or have them respond where the premise explicitly says they stayed silent or did not answer - invent dialogue only to fill genuine gaps the premise leaves open, never to contradict what it states.
- NEVER write a "narrator" line that summarizes, recounts, or describes past events in past tense (e.g. "So I divorced her and stopped listening to her") - every single line must be something a character says TO another character, in the moment, inside a real scene. If the premise is first-person memoir prose, DRAMATIZE it: pick the real scenes it already describes with quoted dialogue, and have the real people in them speak directly to each other there - do not have the narrator summarize what happened between those scenes.
- If the premise contains more real quoted moments than {line_count} lines can hold, you MUST choose which ones to keep - and the choice is not arbitrary: keep the lines the story's actual events depend on (what starts the conflict, each real turning point, how it ends) over lines that are merely additional detail or repeat a point another kept line already makes. Dropping the line that explains WHY something central happens (why a divorce, a decision, an estrangement) in favor of two lines from a less essential moment is wrong even if the total count is still correct."""

_COMPLETE_LINES_SYSTEM_PROMPT = """You are transcribing a real story's own quoted dialogue into a shot list - not summarizing, not selecting, not inventing.

Rules:
- Output ONLY a JSON array of objects, nothing else - no markdown fences, no commentary.
- The premise contains exactly {line_count} directly quoted dialogue moments (text inside «guillemets» or "quotes"). Output EXACTLY {line_count} array elements, one per quoted moment, IN THE SAME ORDER they appear in the premise. Do not skip any, do not merge two into one, do not add a line that is not one of the premise's own quoted moments.
- Each object has exactly these keys:
  - "character": one of these EXACT names (same spelling): {cast}
  - "line": the quoted text for that moment, copied EXACTLY as written in the premise - not a single word paraphrased, shortened, or altered.
  - "action": a short, visual-only English stage direction for this moment (facial expression, gesture, what they're doing) - NEVER repeat or translate the spoken line itself here.
  - "scene": an integer, starting at 0 for the first scene and incrementing by 1 each real time the story's own events move to a genuinely different scene (a real change of location and/or a real time jump between where one quoted moment happens and the next) - every quoted moment that is really part of the same scene as the one before it MUST repeat the SAME scene number. A story that stays in one place throughout is one real scene (every element "scene": 0) - do not invent scene breaks the real story does not have.
  - "location": a short (one sentence, English) description of WHERE this scene physically happens, inferred from the real story - the SAME exact text on every element that shares this scene's number.
  - "mood": a short (a few words, English) emotional/atmospheric description of this real scene - the SAME exact text on every element that shares this scene's number. This becomes the real cue a later music step scores this scene with - never a generic word like "neutral", describe what this specific real scene actually feels like.
- NEVER create or use a "narrator" character - every quoted moment belongs to the real person who said it in the premise."""

# Owner's standing instruction, 2026-10-21 ("اريد الحديث معه كما اتحدث
# معك ... محادثة حرة كاملة مع نقاش وتعديل جماعي للقصة" - I want to talk
# to it like I talk to you, a full free conversation with discussion and
# collaborative editing of the story): the bot used to only understand
# rigid commands (/new, /full, /edit <number> <text>, /caption <numbers>)
# - any other plain message was either silently accumulated as raw story
# text or, after a draft existed, simply had nowhere to go. This routes
# ANY free message into one of a small, fixed set of real actions, never
# arbitrary code - the owner can describe what they want in their own
# words and the bot figures out whether that is more story material, a
# request to build the screenplay now, an edit to a specific existing
# line, or a request to caption a line, and replies conversationally
# either way.
#
# Real safety line that must never move: "edit_lines" can still only
# ever change `DialogueLine.line` text through the exact same mechanism
# `/edit` already used (one owner-approved screenplay, read/edited
# before any real money is spent) - and real production (spending real
# money) is NEVER triggered from this router, under any intent, however
# the owner phrases it. It only ever starts from the explicit ✅ button
# already shown under the screenplay - see BotConversation.handle_message.
_ROUTER_SYSTEM_PROMPT = """You are routing one new Telegram message inside an ongoing real conversation about building a real short film - real characters speaking real dialogue, never a narrator.

You are given, in the user message: whether a screenplay draft currently exists (and its full current content if so), any story text the owner has already typed that has not yet been turned into a screenplay, and the owner's brand-new message.

Output ONLY a JSON object, nothing else - no markdown fences, no commentary:
{
  "intent": "accumulate" | "build_screenplay" | "edit_lines" | "set_captions" | "discuss",
  "reply": "<a short (1-3 sentences), natural reply in the SAME language as the owner's message, to show them right now - this is a real conversation, not a form>",
  "edits": [{"line": <1-based line number>, "text": "<new exact line text>"}],
  "caption_lines": [<1-based line numbers>]
}

Rules for choosing "intent":
- "accumulate": the message is part of the story/idea itself (prose, a scene, dialogue, description) with no instruction attached to act on it right now - acknowledge briefly, take no action. "edits" and "caption_lines" are empty arrays.
- "build_screenplay": the owner is explicitly asking to build or start the screenplay now (e.g. "ابن السيناريو", "جاهز", "كمل", "اصنعه الآن", "build it", "ابدأ").
- "edit_lines": the owner is asking to change specific EXISTING line(s) of a draft that already exists - never valid if no draft exists (choose "discuss" instead and say so). If the owner gives their OWN exact words for the new line (quoted, or clearly dictated verbatim), you MUST use that EXACT text in "text" - never paraphrase, shorten, improve, or correct it, even slightly. Only if they describe a change WITHOUT giving exact words may you write new dialogue that fulfills the request, matching that character's established voice in the draft.
- "set_captions": the owner wants specific existing line(s) shown as on-screen text instead of being spoken aloud - never valid if no draft exists.
- "discuss": anything else - a question, an opinion, small talk, or anything that sounds like approving, producing, publishing, or spending real money.
  - ONLY if the owner's words genuinely sound like approval or "do it now" (e.g. "ابدأ الإنتاج", "وافقت", "do it"): "reply" tells them, plainly, to press the ✅ button themselves - real production ONLY ever starts from that button, never from a message here. Never imply here that anything was approved, produced, or paid for.
  - For every other "discuss" message - a question, confusion, a complaint, small talk, "which button?", "why aren't you answering?", anything that is NOT itself a request to approve or produce - "reply" must directly and specifically answer what the owner actually asked or said. Repeating the "press ✅" reminder when they asked something else entirely is itself a real failure mode that has happened live and made an owner feel unheard - never do it.
  - A screenplay line marked "(نص على الشاشة، غير منطوق)" is an on-screen CAPTION (the owner's own choice, costs $0, see /caption), never a narrator. Do not mention "راوٍ"/narrator in "reply" unless the owner's own new message itself raises it - and when they do, answer plainly and specifically from the real cast and lines shown to you (e.g. name which character, if any, or state clearly that none exists) rather than a generic reassurance.
- Never invent an "edits"/"caption_lines" entry for a line number that does not exist in the current draft shown to you - if the owner refers to a line that is not there, intent is "discuss" and "reply" says so."""

# Matches an Arabic guillemet quote («...») or a straight/curly double-quote
# ("..."), non-greedy so two separate quotes on one line count as two
# moments, not one spanning both.
_QUOTED_MOMENT_PATTERN = re.compile(r"«[^»]+»|\"[^\"]+\"|“[^”]+”")

# Owner's real, repeated complaint, 2026-10-22 - see _write()'s own
# comment at its real use site. 0.7 is a real engineering choice, not a
# theoretical ideal: strict enough to reject the real reported case (one
# voice the entire video, confirmed directly by the owner), loose enough
# that a real scene with one naturally talkative character among several
# genuinely participating ones is not rejected for a plausible,
# legitimate imbalance.
_MAX_SINGLE_CHARACTER_LINE_SHARE = 0.7

# Owner's standing instruction, 2026-10-21: no narrator, ever, unless the
# owner explicitly asks for one - a code-level gate, not a prompt request
# the model can ignore. Matches "راوي"/"الراوي" anywhere in the name (the
# model has cast it as "الراوي الأول", "راوي القصة", etc., not only the
# bare word) and "narrator" case-insensitively.
_NARRATOR_NAME_PATTERN = re.compile(r"راوي|narrator", re.IGNORECASE)


def count_quoted_moments(premise: str) -> int:
    # Real incident, 2026-10-20: a real long story's quoted-line count
    # came back far lower than the real number of quoted moments in it
    # (15 instead of 23), with no error - reproduced here: if even ONE
    # quote mark anywhere in the premise is dropped, mismatched, or
    # converted by a phone's keyboard/clipboard into a different
    # character while pasting, _QUOTED_MOMENT_PATTERN's "[^"]+" keeps
    # matching from the next stray opening quote all the way to whatever
    # closing quote it finds NEXT - silently swallowing every real line
    # in between with no warning at all. This is the exact opposite of
    # the owner's own standing rule ("يجب ان ينفذ كل حرف و كل كلمة كتبتها
    # بدون حذف او نقصان" - every letter and word must be used, nothing
    # dropped) - a silent undercount is worse than a loud error, because
    # nothing in the bot's own reply would ever reveal it happened.
    # An odd count of straight quote marks, or a mismatched guillemet/
    # curly-quote pair count, both PROVE at least one quote is unpaired
    # somewhere - raised here instead of silently losing whatever falls
    # inside the broken span.
    straight = premise.count('"')
    if straight % 2 != 0:
        raise DialogueWriterError(
            f"عدد علامات الاقتباس \" في نصك فردي ({straight} علامة) - هذا يعني أن علامة واحدة "
            "ناقصة أو زائدة في مكان ما، وأي حوار بينها وبين العلامة التالية سيُفقد بصمت بدلاً من "
            "أن يُستخدم. راجع نصك وتأكد أن كل جملة منقولة محاطة بعلامتي اقتباس كاملتين \" \" ثم أعد الإرسال."
        )
    curly_open, curly_close = premise.count("“"), premise.count("”")
    if curly_open != curly_close:
        raise DialogueWriterError(
            f"علامات الاقتباس المنحنية “ ” غير متطابقة في نصك ({curly_open} فتح، {curly_close} إغلاق) - "
            "نفس خطر فقدان حوار بصمت. راجع نصك وأعد الإرسال."
        )
    guillemet_open, guillemet_close = premise.count("«"), premise.count("»")
    if guillemet_open != guillemet_close:
        raise DialogueWriterError(
            f"علامات الاقتباس « » غير متطابقة في نصك ({guillemet_open} فتح، {guillemet_close} إغلاق) - "
            "نفس خطر فقدان حوار بصمت. راجع نصك وأعد الإرسال."
        )
    return len(_QUOTED_MOMENT_PATTERN.findall(premise))


class DialogueWriterError(Exception):
    pass


@dataclass
class RoutedMessage:
    """One free Telegram message, classified into exactly one of a fixed
    set of real actions - see _ROUTER_SYSTEM_PROMPT's own comment for why
    this exists and the safety line it never crosses (real production is
    never one of these intents)."""

    intent: str  # "accumulate" | "build_screenplay" | "edit_lines" | "set_captions" | "discuss"
    reply: str
    edits: list[tuple[int, str]] = field(default_factory=list)
    caption_lines: list[int] = field(default_factory=list)


class DialogueWriterAgent:
    def __init__(self, reasoning_client: AgentReasoningClient):
        self._reasoning_client = reasoning_client

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """Every real call into the reasoning client goes through here,
        never `self._reasoning_client.complete()` directly.

        Real incident, 2026-10-22: a real Gemini 500 ("INTERNAL") reached
        the owner verbatim as `❌ حدث خطأ حقيقي غير متوقع: INTERNAL.
        {'error': {'code': 500, ...}}` - runner.py's generic top-level
        error handler, not the bot's own normal reply path. Both Groq
        AND its Gemini fallback had genuinely failed (RealGeminiReasoningClient's
        own 3 retries exhausted) - a real external-provider outage, not a
        bug in either client. But nothing here ever caught that: only
        JSON-parsing/validation failures were wrapped into
        DialogueWriterError, so a raw network/provider exception
        propagated straight out of route_message()/_write(), past
        conversation.py's `except DialogueWriterError` guards (which
        cannot catch an exception of a different type), all the way to
        the generic handler - English, raw JSON, no indication of what
        to do. Every real call site now goes through this wrapper so any
        failure - whatever its real type - reaches the owner as one
        clean, understandable Arabic message through the bot's normal
        reply path instead."""
        try:
            return self._reasoning_client.complete(system_prompt, user_prompt)
        except DialogueWriterError:
            raise
        except Exception as exc:
            raise DialogueWriterError(
                f"تعذّر الاتصال بالذكاء الاصطناعي حالياً (Groq/Gemini) - حاول مرة أخرى بعد قليل. "
                f"لم يُصرف أي مال حقيقي. السبب التقني: {exc}"
            ) from exc

    def route_message(self, screenplay: DialogueScreenplay | None, pending_text: str, new_message: str) -> RoutedMessage:
        parts: list[str] = []
        if screenplay is not None:
            draft_lines = "\n".join(
                f"{l.index + 1}. {l.character}: {l.line}" + ("" if l.spoken else " (نص على الشاشة، غير منطوق)")
                for l in screenplay.lines
            )
            parts.append(f"[مسودة سيناريو موجودة حاليًا]\nالعنوان: {screenplay.title}\n{draft_lines}")
        else:
            parts.append("[لا توجد مسودة سيناريو بعد]")
        if pending_text:
            parts.append(f"[نص قصة مُجمَّع لم يُبنَ منه سيناريو بعد]\n{pending_text}")
        parts.append(f"[رسالة المالك الجديدة]\n{new_message}")
        user_prompt = "\n\n".join(parts)

        raw = self._complete(_ROUTER_SYSTEM_PROMPT, user_prompt)
        data = parse_json_object(raw, ("intent", "reply"), error_cls=DialogueWriterError)

        intent = data.get("intent")
        if intent not in ("accumulate", "build_screenplay", "edit_lines", "set_captions", "discuss"):
            raise DialogueWriterError(f"Router reply had an unrecognized intent {intent!r}: {raw!r}")
        # Real safety check, not just a prompt instruction the model could
        # ignore: edit_lines/set_captions are structurally meaningless
        # without an existing draft - reject here rather than trust the
        # model never to return them anyway (the same code-level-gate
        # principle as the narrator check above).
        if intent in ("edit_lines", "set_captions") and screenplay is None:
            raise DialogueWriterError(
                f"Router returned intent {intent!r} but no screenplay draft exists yet: {raw!r}"
            )

        reply = data.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            raise DialogueWriterError(f"Router reply had no usable 'reply' text: {raw!r}")

        edits: list[tuple[int, str]] = []
        if intent == "edit_lines":
            raw_edits = data.get("edits")
            if not isinstance(raw_edits, list) or not raw_edits:
                raise DialogueWriterError(f"Router intent was edit_lines but gave no real edits: {raw!r}")
            for item in raw_edits:
                if not isinstance(item, dict) or "line" not in item or not isinstance(item.get("text"), str) or not item["text"].strip():
                    raise DialogueWriterError(f"Router edit item missing a real line/text: {item!r}")
                try:
                    line_number = int(item["line"])
                except (TypeError, ValueError):
                    raise DialogueWriterError(f"Router edit item had a non-numeric line: {item!r}")
                edits.append((line_number, item["text"]))

        caption_lines: list[int] = []
        if intent == "set_captions":
            raw_lines = data.get("caption_lines")
            if not isinstance(raw_lines, list) or not raw_lines:
                raise DialogueWriterError(f"Router intent was set_captions but gave no real line numbers: {raw!r}")
            for n in raw_lines:
                try:
                    caption_lines.append(int(n))
                except (TypeError, ValueError):
                    raise DialogueWriterError(f"Router caption_lines item was not a real number: {n!r}")

        return RoutedMessage(intent=intent, reply=reply, edits=edits, caption_lines=caption_lines)

    def write_dialogue_screenplay(self, title: str, premise: str, line_count: int) -> DialogueScreenplay:
        if line_count < 1:
            raise DialogueWriterError(f"line_count must be >= 1, got {line_count}")
        return self._write(title, premise, line_count, _LINES_SYSTEM_PROMPT)

    def write_complete_dialogue_screenplay(self, title: str, premise: str) -> DialogueScreenplay:
        # Real bug found by code review, 2026-10-19: max(1, ...) used to
        # floor a genuinely quote-free premise at 1, which told the
        # model "the premise contains exactly 1 directly quoted moment"
        # when it contains zero - forcing it to invent the one thing
        # /full exists to guarantee never happens.
        line_count = count_quoted_moments(premise)
        if line_count < 1:
            raise DialogueWriterError(
                "لم أجد أي اقتباس حقيقي (بين «» أو \"\") في النص - /full يحتاج كلامًا منقولًا حرفيًا عن الشخصيات لبنائه. "
                "استخدم /new بدلًا منه إن كنت تريد حوارًا جديدًا بلا اقتباسات جاهزة."
            )
        return self._write(title, premise, line_count, _COMPLETE_LINES_SYSTEM_PROMPT)

    def _write(self, title: str, premise: str, line_count: int, lines_system_prompt_template: str) -> DialogueScreenplay:
        cast_raw = self._complete(_CAST_SYSTEM_PROMPT, premise)
        cast = self._parse_cast(cast_raw)
        cast_names = [c["name"] for c in cast]

        lines_system_prompt = lines_system_prompt_template.format(line_count=line_count, cast=", ".join(cast_names))
        lines_raw = self._complete(lines_system_prompt, premise)
        parsed_lines = parse_json_object_array(
            lines_raw, ("character", "line", "action", "scene", "location", "mood"), error_cls=DialogueWriterError
        )
        if len(parsed_lines) < line_count:
            raise DialogueWriterError(f"Reply returned {len(parsed_lines)} lines, needed {line_count}: {lines_raw!r}")
        parsed_lines = parsed_lines[:line_count]

        for item in parsed_lines:
            if item["character"] not in cast_names:
                raise DialogueWriterError(
                    f"Line speaker {item['character']!r} is not in the cast {cast_names}: {lines_raw!r}"
                )
            # Real validation, not just a prompt request the model can
            # ignore: "scene" must be a real non-negative integer, and
            # "location"/"mood" must be real, non-empty text - see this
            # file's own module docstring (owner's 2026-10-23 finding) for
            # why an empty value here is a real gap, not "not set yet".
            if not isinstance(item.get("scene"), int) or item["scene"] < 0:
                raise DialogueWriterError(f"Line item has a non-integer or negative 'scene': {item!r}")
            for key in ("location", "mood"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise DialogueWriterError(f"Line item is missing a real non-empty {key!r}: {item!r}")

        # Owner's real, repeated complaint, 2026-10-22 ("كله الفيديو كله
        # صورة شخص يسرد قصة" - the whole video is one person's photo
        # narrating a story; confirmed directly: "صوت واحد فقط طوال
        # الفيديو" - only one voice the entire video): the existing
        # _NARRATOR_NAME_PATTERN guard below only checks the cast's own
        # NAMES - it says nothing about a real, legitimately-named
        # character being used AS a narrator in practice, carrying
        # nearly the whole screenplay alone while the rest of a real
        # multi-person cast gets little or nothing to say. That is
        # functionally a narrator wearing a human name, and the prompts'
        # own "no narrator" instructions cannot be trusted alone to
        # catch it - a prompt is an instruction the model can still get
        # wrong, the same reason _NARRATOR_NAME_PATTERN exists as a
        # code-level gate rather than a prompt request. Rejected here,
        # deterministically, before any real cost is spent.
        if len(cast_names) > 1:
            line_counts: dict[str, int] = {}
            for item in parsed_lines:
                line_counts[item["character"]] = line_counts.get(item["character"], 0) + 1
            dominant_name, dominant_count = max(line_counts.items(), key=lambda kv: kv[1])
            dominant_share = dominant_count / line_count
            if dominant_share > _MAX_SINGLE_CHARACTER_LINE_SHARE:
                raise DialogueWriterError(
                    f"شخصية واحدة ({dominant_name}) تحمل {dominant_count} من أصل {line_count} سطراً "
                    f"({dominant_share:.0%}) - هذا صوت واحد يسرد القصة عملياً طوال الفيديو، حتى لو لم "
                    "يحمل اسم \"راوي\" حرفياً، وهو بالضبط ما طلبت ألا يحدث أبداً. أعد صياغة الفكرة بحيث "
                    "تتوزع الأحداث كحوار حقيقي بين الشخصيات المختلفة، لا شخصية واحدة تحكي عن البقية."
                )

        characters = [
            Character(
                name=c["name"],
                voice=resolve_placeholder_voice(c["gender"]),
                appearance=c["appearance"],
                gender=c["gender"].strip().lower(),
            )
            for c in cast
        ]
        dialogue_lines = [
            DialogueLine(
                index=i, character=item["character"], line=item["line"], action=item["action"], scene=item["scene"]
            )
            for i, item in enumerate(parsed_lines)
        ]
        scenes = self._build_scenes(parsed_lines)
        return DialogueScreenplay(
            title=title, premise=premise, characters=characters, lines=dialogue_lines, scenes=scenes
        )

    @staticmethod
    def _build_scenes(parsed_lines: list[dict]) -> list[DialogueSceneInfo]:
        """Groups the real parsed lines into real DialogueSceneInfo entries,
        in the order each scene number first appears. `characters_present`
        is derived from which real characters actually speak inside that
        scene's own lines - never asked of the model separately (see this
        file's own module docstring for why a second, model-reported list
        would risk silently drifting from the real lines it describes).
        A scene's `location`/`mood` are taken from its first line - every
        line sharing that scene number is required (above) to repeat the
        same text, so the first occurrence is the real value either way."""
        scenes_by_index: dict[int, DialogueSceneInfo] = {}
        for item in parsed_lines:
            scene_index = item["scene"]
            scene = scenes_by_index.get(scene_index)
            if scene is None:
                scene = DialogueSceneInfo(index=scene_index, location=item["location"], mood=item["mood"])
                scenes_by_index[scene_index] = scene
            if item["character"] not in scene.characters_present:
                scene.characters_present.append(item["character"])
        return [scenes_by_index[i] for i in sorted(scenes_by_index)]

    @staticmethod
    def _parse_cast(raw_reply: str) -> list[dict]:
        text = raw_reply.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            # Real incident, caught live, 2026-10-21: when the premise has
            # no real story content (e.g. an instruction sentence with
            # nothing to cast, sent after the owner's pending story text
            # was lost to a container restart), the casting model does not
            # error - it replies in plain English asking what the story or
            # characters are. The raw repr of that reply used to reach the
            # owner verbatim ("Cast reply was not valid JSON: 'What is the
            # story premise...'") - confusing, in the wrong language, and
            # gave no indication of what to actually do.
            raise DialogueWriterError(
                "تعذّر توليد قائمة الشخصيات - النص المُرسل لا يحتوي على قصة أو حوار حقيقي كافٍ ليُستخرج منه شخصيات "
                f"حقيقية. أرسل القصة كاملة كرسالة عادية أولاً، ثم اطلب بناء السيناريو. (رد النموذج: {raw_reply!r})"
            ) from exc
        if not isinstance(parsed, list) or not parsed:
            raise DialogueWriterError(
                "تعذّر توليد قائمة الشخصيات - النموذج لم يجد أي شخصية حقيقية تتكلم في النص المُرسل. "
                f"أرسل القصة كاملة أولاً، أو تأكد أنها تحتوي على شخصيات تتحدث فعلاً. (رد النموذج: {raw_reply!r})"
            )
        for item in parsed:
            # Real bug found by code review, 2026-10-19: this only
            # checked the keys existed, never that their values were
            # real non-empty strings. A model reply like {"name": "X",
            # "gender": null} passed this check, then crashed deeper
            # with an uncaught AttributeError inside character_photos.py's
            # None.strip() instead of a clear error here - exactly the
            # kind of failure the global error handler (runner.py) now
            # at least reports, but this is the right place to catch it
            # with a real, specific message.
            if not isinstance(item, dict) or not all(
                isinstance(item.get(k), str) and item.get(k).strip()
                for k in ("name", "gender", "appearance")
            ):
                raise DialogueWriterError(
                    f"Cast reply item {item!r} is missing or has a non-string/empty "
                    f"'name'/'gender'/'appearance': {raw_reply!r}"
                )
            if _NARRATOR_NAME_PATTERN.search(item["name"]):
                raise DialogueWriterError(
                    f"الذكاء الاصطناعي حاول إضافة \"{item['name']}\" كراوي - هذا ممنوع نهائيًا. كل من "
                    "يتكلم في القصة يجب أن يكون شخصية حقيقية تظهر في المشهد وتتكلم مباشرة، لا راويًا "
                    f"يسرد الأحداث. أعد المحاولة أو عدّل النص: {raw_reply!r}"
                )
        # Real bug found by code review, 2026-10-21: two cast entries
        # with the same name were never rejected. DialoguePipeline keys
        # its own character lookup by name (characters_by_name = {c.name:
        # c for c in screenplay.characters}) - a dict comprehension keeps
        # only the LAST entry for a duplicate key, silently discarding
        # the first one's gender/voice/appearance. Any line attributed to
        # that name would then silently use the wrong character's real
        # generated photo and voice - the exact "wrong face" defect this
        # whole project has been fighting, through a different door.
        names = [item["name"] for item in parsed]
        if len(names) != len(set(names)):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise DialogueWriterError(
                f"الذكاء الاصطناعي كرر نفس اسم الشخصية أكثر من مرة ({', '.join(duplicates)}) - هذا يسبب "
                f"اختلاط الأصوات والصور بين شخصيتين مختلفتين. أعد المحاولة أو عدّل النص: {raw_reply!r}"
            )
        return parsed
