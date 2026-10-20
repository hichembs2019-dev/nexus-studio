"""Dialogue Writer Agent: turns a one-line idea into a real character
screenplay - actual people speaking actual lines - the owner's real ask
("مسلسلات وأفلام ... لربح المال", series and films, not a narrated
documentary). Replaces WriterAgent's prose-beat output for any episode
meant to go through DialoguePipeline (real lip-synced speech, see
fal_video_client.py): that pipeline needs a speaker attached to every
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
حسب وصف كل شخصية" - the photos are of course placeholders, assigned
automatically per character description): the cast step now also infers
each character's gender/age_group, and every Character is built with a
real reference_image_path and voice already assigned (see
character_photos.py) - never the empty string DialoguePipeline refuses
to accept.

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
"""
from __future__ import annotations

import json
import re

from src.agents._text_parsing import parse_json_object_array
from src.integrations.reasoning_client import AgentReasoningClient
from src.models import Character, DialogueLine, DialogueScreenplay
from src.workflows.character_photos import resolve_placeholder_photo, resolve_placeholder_voice

_CAST_SYSTEM_PROMPT = """You are a casting director listing the speaking characters a short story needs.

Rules:
- Output ONLY a JSON array of objects, nothing else - no markdown fences, no commentary.
- Each object has exactly four keys:
  - "name": the character's name, as it should appear consistently everywhere (same spelling every time).
  - "gender": "male" or "female" - your best real inference from how the premise describes or refers to them.
  - "age_group": "young" (child/teen/20s), "middle" (30s-50s), or "older" (60+) - your best real inference.
  - "appearance": a short (one sentence, English) physical description for generating a reference photo of THIS character - hair color/style, build, any distinguishing feature the premise mentions. If the premise gives no physical detail, invent a plausible, ordinary one consistent with their gender/age/role - never leave this generic or reuse the exact same wording for two different characters in the same cast.
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
- Each object has exactly three keys:
  - "character": one of these EXACT names (same spelling): {cast}
  - "line": what they actually SAY, in the SAME language as the premise, 1-2 short sentences a person would plausibly speak aloud - never a narrator's description of events.
  - "action": a short, visual-only English stage direction for this moment (facial expression, gesture, what they're doing) - NEVER repeat or translate the spoken line itself here, the video model lip-syncs the real audio and would render duplicated text as more speech.
- The lines must flow as one continuous scene in order, matching the premise, with characters actually responding to each other.
- The premise is the real source of truth, not a loose inspiration: if it already contains a character's actual words (quoted directly, e.g. in "quotes" or «guillemets»), use that EXACT wording for their line - do not paraphrase or invent a different line for a moment the premise already gives you.
- Never give a character a line the premise explicitly says they did NOT say, or have them respond where the premise explicitly says they stayed silent or did not answer - invent dialogue only to fill genuine gaps the premise leaves open, never to contradict what it states.
- NEVER write a "narrator" line that summarizes, recounts, or describes past events in past tense (e.g. "So I divorced her and stopped listening to her") - every single line must be something a character says TO another character, in the moment, inside a real scene. If the premise is first-person memoir prose, DRAMATIZE it: pick the real scenes it already describes with quoted dialogue, and have the real people in them speak directly to each other there - do not have the narrator summarize what happened between those scenes.
- If the premise contains more real quoted moments than {line_count} lines can hold, you MUST choose which ones to keep - and the choice is not arbitrary: keep the lines the story's actual events depend on (what starts the conflict, each real turning point, how it ends) over lines that are merely additional detail or repeat a point another kept line already makes. Dropping the line that explains WHY something central happens (why a divorce, a decision, an estrangement) in favor of two lines from a less essential moment is wrong even if the total count is still correct."""

_COMPLETE_LINES_SYSTEM_PROMPT = """You are transcribing a real story's own quoted dialogue into a shot list - not summarizing, not selecting, not inventing.

Rules:
- Output ONLY a JSON array of objects, nothing else - no markdown fences, no commentary.
- The premise contains exactly {line_count} directly quoted dialogue moments (text inside «guillemets» or "quotes"). Output EXACTLY {line_count} array elements, one per quoted moment, IN THE SAME ORDER they appear in the premise. Do not skip any, do not merge two into one, do not add a line that is not one of the premise's own quoted moments.
- Each object has exactly three keys:
  - "character": one of these EXACT names (same spelling): {cast}
  - "line": the quoted text for that moment, copied EXACTLY as written in the premise - not a single word paraphrased, shortened, or altered.
  - "action": a short, visual-only English stage direction for this moment (facial expression, gesture, what they're doing) - NEVER repeat or translate the spoken line itself here.
- NEVER create or use a "narrator" character - every quoted moment belongs to the real person who said it in the premise."""

# Matches an Arabic guillemet quote («...») or a straight/curly double-quote
# ("..."), non-greedy so two separate quotes on one line count as two
# moments, not one spanning both.
_QUOTED_MOMENT_PATTERN = re.compile(r"«[^»]+»|\"[^\"]+\"|“[^”]+”")


def _count_quoted_moments(premise: str) -> int:
    return len(_QUOTED_MOMENT_PATTERN.findall(premise))


class DialogueWriterError(Exception):
    pass


class DialogueWriterAgent:
    def __init__(self, reasoning_client: AgentReasoningClient):
        self._reasoning_client = reasoning_client

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
        line_count = _count_quoted_moments(premise)
        if line_count < 1:
            raise DialogueWriterError(
                "لم أجد أي اقتباس حقيقي (بين «» أو \"\") في النص - /full يحتاج كلامًا منقولًا حرفيًا عن الشخصيات لبنائه. "
                "استخدم /new بدلًا منه إن كنت تريد حوارًا جديدًا بلا اقتباسات جاهزة."
            )
        return self._write(title, premise, line_count, _COMPLETE_LINES_SYSTEM_PROMPT)

    def _write(self, title: str, premise: str, line_count: int, lines_system_prompt_template: str) -> DialogueScreenplay:
        cast_raw = self._reasoning_client.complete(_CAST_SYSTEM_PROMPT, premise)
        cast = self._parse_cast(cast_raw)
        cast_names = [c["name"] for c in cast]

        lines_system_prompt = lines_system_prompt_template.format(line_count=line_count, cast=", ".join(cast_names))
        lines_raw = self._reasoning_client.complete(lines_system_prompt, premise)
        parsed_lines = parse_json_object_array(
            lines_raw, ("character", "line", "action"), error_cls=DialogueWriterError
        )
        if len(parsed_lines) < line_count:
            raise DialogueWriterError(f"Reply returned {len(parsed_lines)} lines, needed {line_count}: {lines_raw!r}")

        for item in parsed_lines[:line_count]:
            if item["character"] not in cast_names:
                raise DialogueWriterError(
                    f"Line speaker {item['character']!r} is not in the cast {cast_names}: {lines_raw!r}"
                )

        characters = [
            Character(
                name=c["name"],
                # The cheap generic placeholder, used only as a free
                # preview before Approve and as DialoguePipeline's fallback
                # if real portrait generation fails - DialoguePipeline
                # replaces this with a real, story-specific photo
                # generated from `appearance` once production is approved.
                reference_image_path=resolve_placeholder_photo(c["gender"], c["age_group"]),
                voice=resolve_placeholder_voice(c["gender"]),
                appearance=c["appearance"],
            )
            for c in cast
        ]
        dialogue_lines = [
            DialogueLine(index=i, character=item["character"], line=item["line"], action=item["action"])
            for i, item in enumerate(parsed_lines[:line_count])
        ]
        return DialogueScreenplay(title=title, premise=premise, characters=characters, lines=dialogue_lines)

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
            raise DialogueWriterError(f"Cast reply was not valid JSON: {raw_reply!r}") from exc
        if not isinstance(parsed, list) or not parsed:
            raise DialogueWriterError(f"Cast reply was not a non-empty JSON array: {raw_reply!r}")
        for item in parsed:
            # Real bug found by code review, 2026-10-19: this only
            # checked the keys existed, never that their values were
            # real non-empty strings. A model reply like {"name": "X",
            # "gender": null, "age_group": "middle"} passed this check,
            # then crashed deeper with an uncaught AttributeError inside
            # character_photos.py's None.strip() instead of a clear
            # error here - exactly the kind of failure the global error
            # handler (runner.py) now at least reports, but this is the
            # right place to catch it with a real, specific message.
            if not isinstance(item, dict) or not all(
                isinstance(item.get(k), str) and item.get(k).strip()
                for k in ("name", "gender", "age_group", "appearance")
            ):
                raise DialogueWriterError(
                    f"Cast reply item {item!r} is missing or has a non-string/empty "
                    f"'name'/'gender'/'age_group'/'appearance': {raw_reply!r}"
                )
        return parsed
