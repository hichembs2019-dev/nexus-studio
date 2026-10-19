from __future__ import annotations

import json

from src.agents.dialogue_writer import DialogueWriterAgent
from src.agents.publisher import PublishingAgent
from src.bot.conversation import ESTIMATED_SECONDS_PER_LINE, BotConversation
from src.integrations.publishing_client import MockPublishingClient
from src.integrations.reasoning_client import MockAgentReasoningClient
from src.models import Character, DialogueLine, DialogueScreenplay, ProducedDialogueClip, ProducedDialogueEpisode


def _cast_reply(*people):
    return json.dumps([{"name": n, "gender": g, "age_group": a} for n, g, a in people])


def _lines_reply(*lines):
    return json.dumps([{"character": c, "line": l, "action": a} for c, l, a in lines])


def _conversation(reasoning_replies: list[str], youtube_video_ids: list[str] | None = None) -> tuple[BotConversation, MockPublishingClient]:
    reasoning = MockAgentReasoningClient(scripted_replies=reasoning_replies)
    youtube = MockPublishingClient(scripted_video_ids=youtube_video_ids or ["yt-id"])
    conversation = BotConversation(DialogueWriterAgent(reasoning), PublishingAgent({"youtube": youtube}))
    return conversation, youtube


def test_new_with_explicit_seconds_writes_a_screenplay_sized_to_them():
    cast = _cast_reply(("فوزية", "female", "middle"))
    lines = _lines_reply(("فوزية", "beat one", "smiling"), ("فوزية", "beat two", "waving"))
    conversation, _ = _conversation([cast, lines])

    reply = conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} A lone astronaut finds a signal.")

    assert "beat one" in reply.text
    assert "beat two" in reply.text
    assert reply.buttons == [("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")]


def test_new_without_a_leading_number_uses_the_default_duration_and_whole_text_as_premise():
    from src.bot.conversation import DEFAULT_TARGET_SECONDS

    line_count = max(2, round(DEFAULT_TARGET_SECONDS / ESTIMATED_SECONDS_PER_LINE))
    cast = _cast_reply(("فوزية", "female", "middle"))
    lines = _lines_reply(*[("فوزية", f"b{i}", "") for i in range(line_count)])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast, lines])
    conversation = BotConversation(DialogueWriterAgent(reasoning), PublishingAgent({}))

    conversation.handle_new("chat1", "A lone astronaut finds a signal.")

    assert reasoning.calls[0][1] == "A lone astronaut finds a signal."


def test_new_with_empty_text_asks_for_an_idea_without_calling_the_writer():
    conversation, _ = _conversation([])

    reply = conversation.handle_new("chat1", "   ")

    assert "الاستخدام" in reply.text


def test_plain_text_messages_are_accumulated_and_new_without_args_uses_them_verbatim():
    # Real finding, 2026-10-19: Telegram auto-splits a long outgoing
    # message (~900 real words) into several plain messages, and only the
    # first carries /new - the rest must still reach the writer, whole,
    # not summarized.
    cast = _cast_reply(("فوزية", "female", "middle"))
    lines = _lines_reply(("فوزية", "beat one", ""), ("فوزية", "beat two", ""))
    conversation, _ = _conversation([cast, lines])

    conversation.handle_plain_text("chat1", "الجزء الأول من القصة الطويلة")
    conversation.handle_plain_text("chat1", "والجزء الثاني الذي وصل في رسالة منفصلة")
    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)}")

    # the writer's second argument on each call is the premise it was given
    sent_premise = conversation._dialogue_writer._reasoning_client.calls[0][1]
    assert "الجزء الأول من القصة الطويلة" in sent_premise
    assert "والجزء الثاني الذي وصل في رسالة منفصلة" in sent_premise


def test_plain_text_accumulation_is_cleared_after_a_screenplay_is_built():
    cast1 = _cast_reply(("فوزية", "female", "middle"))
    lines1 = _lines_reply(("فوزية", "beat one", ""), ("فوزية", "beat two", ""))
    cast2 = _cast_reply(("جويدة", "female", "young"))
    lines2 = _lines_reply(("جويدة", "new idea beat", ""), ("جويدة", "new idea beat two", ""))
    conversation, _ = _conversation([cast1, lines1, cast2, lines2])

    conversation.handle_plain_text("chat1", "القصة القديمة")
    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)}")
    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} فكرة جديدة تمامًا")

    assert "القصة القديمة" not in conversation._dialogue_writer._reasoning_client.calls[2][1]


def test_cancel_clears_pending_text_too():
    conversation, _ = _conversation([])

    conversation.handle_plain_text("chat1", "نص معلّق")
    conversation.handle_cancel("chat1")
    reply = conversation.handle_new("chat1", "")

    assert "الاستخدام" in reply.text


def test_full_builds_a_screenplay_with_no_duration_and_one_line_per_real_quote():
    # Owner's instruction, 2026-10-19 ("يجب ان ينفذ كل حرف و كل كلمة
    # كتبتها بدون حذف او نقصان" - every letter and word must be used, no
    # deletion, no shortening): /full has no seconds argument and never
    # estimates a line_count - it uses exactly as many lines as the real
    # story has quoted moments.
    premise = 'قال: «مرحبا يا فوزية» فقالت: «أهلا بك»'
    cast = _cast_reply(("الرجل", "male", "middle"), ("فوزية", "female", "middle"))
    lines = _lines_reply(("الرجل", "مرحبا يا فوزية", ""), ("فوزية", "أهلا بك", ""))
    conversation, _ = _conversation([cast, lines])

    reply = conversation.handle_full("chat1", premise)

    assert "مرحبا يا فوزية" in reply.text
    assert "أهلا بك" in reply.text
    assert reply.buttons == [("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")]


def test_full_uses_accumulated_plain_text_when_no_args_given():
    premise = 'قال: «مرحبا يا فوزية» فقالت: «أهلا بك»'
    cast = _cast_reply(("الرجل", "male", "middle"), ("فوزية", "female", "middle"))
    lines = _lines_reply(("الرجل", "مرحبا يا فوزية", ""), ("فوزية", "أهلا بك", ""))
    conversation, _ = _conversation([cast, lines])

    conversation.handle_plain_text("chat1", premise)
    reply = conversation.handle_full("chat1", "")

    assert "مرحبا يا فوزية" in reply.text


def test_full_with_no_text_at_all_asks_for_the_story_without_calling_the_writer():
    conversation, _ = _conversation([])

    reply = conversation.handle_full("chat1", "   ")

    assert "الاستخدام" in reply.text


def test_edit_updates_the_named_line_and_resends_the_screenplay():
    cast = _cast_reply(("فوزية", "female", "middle"))
    lines = _lines_reply(("فوزية", "beat one", ""), ("فوزية", "beat two", ""))
    conversation, _ = _conversation([cast, lines])
    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea")

    reply = conversation.handle_edit("chat1", "1 a much better beat")

    assert "a much better beat" in reply.text
    assert "beat one" not in reply.text
    assert "beat two" in reply.text


def test_edit_on_an_out_of_range_line_reports_it_clearly():
    cast = _cast_reply(("فوزية", "female", "middle"))
    lines = _lines_reply(("فوزية", "beat one", ""), ("فوزية", "beat two", ""))
    conversation, _ = _conversation([cast, lines])
    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea")

    reply = conversation.handle_edit("chat1", "5 something")

    assert "سطر رقم 5" in reply.text


def test_edit_without_a_draft_tells_the_user_to_start_with_new():
    conversation, _ = _conversation([])

    reply = conversation.handle_edit("chat1", "1 x")

    assert "/new" in reply.text


def test_approve_returns_the_exact_draft_screenplay_for_real_production():
    cast = _cast_reply(("فوزية", "female", "middle"))
    lines = _lines_reply(("فوزية", "beat one", ""), ("فوزية", "beat two", ""))
    conversation, _ = _conversation([cast, lines])
    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea")

    reply, screenplay = conversation.handle_approve("chat1")

    assert screenplay is not None
    assert screenplay.lines[0].line == "beat one"
    assert "2" in reply.text  # mentions the real line count being generated


def test_approve_without_a_draft_returns_none_screenplay():
    conversation, _ = _conversation([])

    reply, screenplay = conversation.handle_approve("chat1")

    assert screenplay is None
    assert "/new" in reply.text


def test_format_produced_offers_publish_buttons_when_stitched():
    conversation, _ = _conversation([])
    screenplay = DialogueScreenplay(title="T", premise="p", characters=[Character(name="X", reference_image_path="/x.png")], lines=[])
    clip = ProducedDialogueClip(line=DialogueLine(index=0, character="X", line="hi"), local_path="/tmp/c0.mp4", cost_usd=0.4)
    produced = ProducedDialogueEpisode(screenplay=screenplay, clips=[clip], stitched_path="/tmp/out.mp4")

    reply = conversation.format_produced("chat1", produced)

    assert reply.video_path == "/tmp/out.mp4"
    assert reply.buttons == [("📤 نشر على YouTube", "publish:youtube"), ("🗑️ تجاهل", "discard")]
    assert "0.400" in reply.text


def test_format_produced_without_a_stitched_path_reports_cost_without_buttons():
    conversation, _ = _conversation([])
    screenplay = DialogueScreenplay(title="T", premise="p", characters=[], lines=[])
    clip = ProducedDialogueClip(line=DialogueLine(index=0, character="X", line="hi"), local_path="/tmp/c0.mp4", cost_usd=0.4)
    produced = ProducedDialogueEpisode(screenplay=screenplay, clips=[clip], stitched_path=None)

    reply = conversation.format_produced("chat1", produced)

    assert reply.video_path is None
    assert reply.buttons is None
    assert "تعذّر دمجها" in reply.text


def test_publish_youtube_success_reports_the_real_url():
    conversation, youtube = _conversation([], youtube_video_ids=["abc123"])
    screenplay = DialogueScreenplay(title="T", premise="p", characters=[], lines=[])
    produced = ProducedDialogueEpisode(screenplay=screenplay, clips=[], stitched_path="/tmp/out.mp4")
    conversation.format_produced("chat1", produced)

    reply = conversation.handle_publish("chat1", "youtube")

    assert reply.text == "✅ نُشر على قناة Nexora: https://youtu.be/abc123"
    assert youtube.uploads[0]["video_path"] == "/tmp/out.mp4"


def test_publish_with_nothing_produced_reports_that():
    conversation, _ = _conversation([])

    reply = conversation.handle_publish("chat1", "youtube")

    assert "لا يوجد شيء جاهز للنشر" in reply.text


def test_publish_to_unconfigured_platform_reports_the_failure():
    conversation, _ = _conversation([])
    screenplay = DialogueScreenplay(title="T", premise="p", characters=[], lines=[])
    produced = ProducedDialogueEpisode(screenplay=screenplay, clips=[], stitched_path="/tmp/out.mp4")
    conversation.format_produced("chat1", produced)

    reply = conversation.handle_publish("chat1", "tiktok")

    assert "فشل النشر" in reply.text
    assert "No client configured" in reply.text


def test_cancel_clears_the_draft_so_approve_then_fails():
    cast = _cast_reply(("فوزية", "female", "middle"))
    lines = _lines_reply(("فوزية", "beat one", ""), ("فوزية", "beat two", ""))
    conversation, _ = _conversation([cast, lines])
    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea")

    conversation.handle_cancel("chat1")
    reply, screenplay = conversation.handle_approve("chat1")

    assert screenplay is None
    assert "/new" in reply.text


def test_chats_are_independent():
    cast1 = _cast_reply(("فوزية", "female", "middle"))
    lines1 = _lines_reply(("فوزية", "chat1 beat", ""), ("فوزية", "chat1 beat two", ""))
    cast2 = _cast_reply(("جويدة", "female", "young"))
    lines2 = _lines_reply(("جويدة", "chat2 beat", ""), ("جويدة", "chat2 beat two", ""))
    conversation, _ = _conversation([cast1, lines1, cast2, lines2])

    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea one")
    conversation.handle_new("chat2", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea two")

    reply1 = conversation.handle_edit("chat1", "1 x")
    assert "chat2 beat" not in reply1.text
