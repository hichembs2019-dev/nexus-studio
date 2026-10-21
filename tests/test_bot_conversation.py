from __future__ import annotations

import json

from src.agents.dialogue_writer import DialogueWriterAgent
from src.agents.publisher import PublishingAgent
from src.bot.conversation import ESTIMATED_SECONDS_PER_LINE, BotConversation, TELEGRAM_MESSAGE_LIMIT, split_for_telegram
from src.integrations.fal_video_client import PRICE_PER_SECOND_USD
from src.integrations.publishing_client import MockPublishingClient
from src.integrations.reasoning_client import MockAgentReasoningClient
from src.models import Character, DialogueLine, DialogueScreenplay, ProducedDialogueClip, ProducedDialogueEpisode


def _cast_reply(*people):
    return json.dumps(
        [{"name": n, "gender": g, "appearance": f"{g}, ordinary appearance"} for n, g in people]
    )


def _lines_reply(*lines):
    return json.dumps([{"character": c, "line": l, "action": a} for c, l, a in lines])


def _conversation(reasoning_replies: list[str], youtube_video_ids: list[str] | None = None) -> tuple[BotConversation, MockPublishingClient]:
    reasoning = MockAgentReasoningClient(scripted_replies=reasoning_replies)
    youtube = MockPublishingClient(scripted_video_ids=youtube_video_ids or ["yt-id"])
    conversation = BotConversation(DialogueWriterAgent(reasoning), PublishingAgent({"youtube": youtube}))
    return conversation, youtube


def test_new_with_explicit_seconds_writes_a_screenplay_sized_to_them():
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "beat one", "smiling"), ("فوزية", "beat two", "waving"))
    conversation, _ = _conversation([cast, lines])

    reply = conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} A lone astronaut finds a signal.")

    assert "beat one" in reply.text
    assert "beat two" in reply.text
    assert reply.buttons == [("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")]


def test_new_without_a_leading_number_uses_the_default_duration_and_whole_text_as_premise():
    from src.bot.conversation import DEFAULT_TARGET_SECONDS

    line_count = max(2, round(DEFAULT_TARGET_SECONDS / ESTIMATED_SECONDS_PER_LINE))
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(*[("فوزية", f"b{i}", "") for i in range(line_count)])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast, lines])
    conversation = BotConversation(DialogueWriterAgent(reasoning), PublishingAgent({}))

    conversation.handle_new("chat1", "A lone astronaut finds a signal.")

    assert reasoning.calls[0][1] == "A lone astronaut finds a signal."


def test_new_with_empty_text_asks_for_an_idea_without_calling_the_writer():
    conversation, _ = _conversation([])

    reply = conversation.handle_new("chat1", "   ")

    assert "الاستخدام" in reply.text


def test_new_refuses_when_the_premise_has_more_real_quotes_than_its_line_count_without_calling_the_writer():
    # Real incident, 2026-10-21: a real 25-real-quote story was sent via
    # /new (default 60s -> 15 lines) instead of /full, which let the
    # model SELECT/invent rather than guarantee every quote verbatim -
    # the real $6.838 wrong video, published live to a real public
    # YouTube URL, is what that cost. /new must refuse BEFORE calling
    # the writer at all (reasoning.calls stays empty - zero real cost
    # risked), not after, and must point the owner at /full.
    conversation, _ = _conversation([])
    premise = " ".join(f'"سطر رقم {i}"' for i in range(16))  # 16 real quotes, default /new -> 15 lines

    reply = conversation.handle_new("chat1", premise)

    assert "/full" in reply.text
    assert "16" in reply.text
    assert not reply.buttons


def test_new_allows_a_premise_whose_real_quote_count_fits_its_line_count():
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "one", ""), ("فوزية", "two", ""))
    conversation, _ = _conversation([cast, lines])
    premise = f'{int(2 * ESTIMATED_SECONDS_PER_LINE)} "واحد" "اثنان"'  # 2 real quotes, 2 lines - exactly fits

    reply = conversation.handle_new("chat1", premise)

    assert reply.buttons == [("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")]


def test_plain_text_messages_are_accumulated_and_new_without_args_uses_them_verbatim():
    # Real finding, 2026-10-19: Telegram auto-splits a long outgoing
    # message (~900 real words) into several plain messages, and only the
    # first carries /new - the rest must still reach the writer, whole,
    # not summarized.
    cast = _cast_reply(("فوزية", "female"))
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
    cast1 = _cast_reply(("فوزية", "female"))
    lines1 = _lines_reply(("فوزية", "beat one", ""), ("فوزية", "beat two", ""))
    cast2 = _cast_reply(("جويدة", "female"))
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
    cast = _cast_reply(("الرجل", "male"), ("فوزية", "female"))
    lines = _lines_reply(("الرجل", "مرحبا يا فوزية", ""), ("فوزية", "أهلا بك", ""))
    conversation, _ = _conversation([cast, lines])

    reply = conversation.handle_full("chat1", premise)

    assert "مرحبا يا فوزية" in reply.text
    assert "أهلا بك" in reply.text
    assert reply.buttons == [("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")]


def test_full_uses_accumulated_plain_text_when_no_args_given():
    premise = 'قال: «مرحبا يا فوزية» فقالت: «أهلا بك»'
    cast = _cast_reply(("الرجل", "male"), ("فوزية", "female"))
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
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "beat one", ""), ("فوزية", "beat two", ""))
    conversation, _ = _conversation([cast, lines])
    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea")

    reply = conversation.handle_edit("chat1", "1 a much better beat")

    assert "a much better beat" in reply.text
    assert "beat one" not in reply.text
    assert "beat two" in reply.text


def test_edit_on_an_out_of_range_line_reports_it_clearly():
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "beat one", ""), ("فوزية", "beat two", ""))
    conversation, _ = _conversation([cast, lines])
    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea")

    reply = conversation.handle_edit("chat1", "5 something")

    assert "لم توجد هذه الأرقام" in reply.text
    assert "5" in reply.text


def test_edit_without_a_draft_tells_the_user_to_start_with_new():
    conversation, _ = _conversation([])

    reply = conversation.handle_edit("chat1", "1 x")

    assert "/new" in reply.text


def test_edit_updates_several_lines_from_one_multiline_message():
    # Owner's instruction, 2026-10-20 ("لماذا كتبت يعدل مشهد واحد؟ بل
    # يعدل كل المشاهد فالحرية لي" - why did you write it edits one
    # scene? it should edit ALL scenes, the freedom is mine): one /edit
    # message, several "<number> <text>" edits, one per real line,
    # applied together in a single pass.
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "beat one", ""), ("فوزية", "beat two", ""), ("فوزية", "beat three", ""))
    conversation, _ = _conversation([cast, lines])
    conversation.handle_new("chat1", f"{int(3 * ESTIMATED_SECONDS_PER_LINE)} idea")

    reply = conversation.handle_edit("chat1", "1 new first line\n3 new third line")

    assert "new first line" in reply.text
    assert "beat one" not in reply.text
    assert "beat two" in reply.text  # untouched line 2 survives
    assert "new third line" in reply.text
    assert "beat three" not in reply.text


def test_edit_reports_not_found_lines_but_still_applies_the_valid_ones():
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "beat one", ""), ("فوزية", "beat two", ""))
    conversation, _ = _conversation([cast, lines])
    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea")

    reply = conversation.handle_edit("chat1", "1 fixed line\n9 does not exist")

    assert "fixed line" in reply.text
    assert "لم توجد هذه الأرقام" in reply.text
    assert "9" in reply.text


def test_edit_with_blank_lines_in_the_message_ignores_them():
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "beat one", ""), ("فوزية", "beat two", ""))
    conversation, _ = _conversation([cast, lines])
    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea")

    reply = conversation.handle_edit("chat1", "1 fixed\n\n   \n")

    assert "fixed" in reply.text
    assert reply.buttons == [("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")]


def test_caption_marks_a_line_as_on_screen_text_with_zero_estimated_cost():
    # Owner's instruction, 2026-10-20 ("نحذف الام و يبقى حوارها مكتوب على
    # الشاشة ... لكي نستحقظ على المال" - remove the mother as a speaking
    # role, her dialogue stays written on screen, to save money):
    # /caption flips a line to spoken=False, which DialoguePipeline then
    # produces for real zero cost - the owner must see this reflected in
    # the estimate BEFORE approving, not discover it only afterward.
    cast = _cast_reply(("فوزية", "female"), ("جويدة", "female"))
    lines = _lines_reply(("فوزية", "beat one", ""), ("جويدة", "beat two", ""))
    conversation, _ = _conversation([cast, lines])
    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea")

    reply = conversation.handle_caption("chat1", "2")

    assert "نص على الشاشة" in reply.text
    # Only line 1 (فوزية) is still spoken/billed - the estimate must drop
    # to reflect just that one line, not both.
    expected_cost = 1 * ESTIMATED_SECONDS_PER_LINE * PRICE_PER_SECOND_USD["480p"]
    assert f"${expected_cost:.2f}" in reply.text
    assert reply.buttons == [("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")]


def test_caption_on_an_out_of_range_line_reports_it_clearly():
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "beat one", ""), ("فوزية", "beat two", ""))
    conversation, _ = _conversation([cast, lines])
    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea")

    reply = conversation.handle_caption("chat1", "9")

    assert "لم توجد هذه الأرقام" in reply.text
    assert "9" in reply.text


def test_caption_without_a_draft_tells_the_user_to_start_with_new():
    conversation, _ = _conversation([])

    reply = conversation.handle_caption("chat1", "1")

    assert "/new" in reply.text


def test_caption_with_no_numbers_shows_usage():
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "beat one", ""), ("فوزية", "beat two", ""))
    conversation, _ = _conversation([cast, lines])
    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea")

    reply = conversation.handle_caption("chat1", "")

    assert "/caption" in reply.text


def test_approve_returns_the_exact_draft_screenplay_for_real_production():
    cast = _cast_reply(("فوزية", "female"))
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
    screenplay = DialogueScreenplay(title="T", premise="p", characters=[Character(name="X")], lines=[])
    clip = ProducedDialogueClip(line=DialogueLine(index=0, character="X", line="hi"), local_path="/tmp/c0.mp4", cost_usd=0.4)
    produced = ProducedDialogueEpisode(screenplay=screenplay, clips=[clip], stitched_path="/tmp/out.mp4")

    reply = conversation.format_produced("chat1", produced)

    assert reply.video_path == "/tmp/out.mp4"
    assert reply.buttons == [("📤 نشر على YouTube", "publish:youtube"), ("🗑️ تجاهل", "discard")]
    assert "0.400" in reply.text


def test_format_produced_without_a_stitched_path_still_delivers_the_real_paid_clips():
    # Real incident, 2026-10-20: a real $8.18 run's clips all genuinely
    # finished, but the merge step failed, and the owner received NEITHER
    # the stitched video NOR the individual clips it already paid for -
    # this test used to assert exactly that (video_path is None, no video
    # at all) as the CORRECT behavior. It is not: real, already-paid
    # clips must reach the owner even when merging into one file fails.
    conversation, _ = _conversation([])
    screenplay = DialogueScreenplay(title="T", premise="p", characters=[], lines=[])
    clip = ProducedDialogueClip(line=DialogueLine(index=0, character="X", line="hi"), local_path="/tmp/c0.mp4", cost_usd=0.4)
    produced = ProducedDialogueEpisode(
        screenplay=screenplay, clips=[clip], stitched_path=None, stitch_error="ffmpeg dialogue concat failed: real reason"
    )

    reply = conversation.format_produced("chat1", produced)

    assert reply.video_path is None
    assert reply.video_paths == ["/tmp/c0.mp4"]
    assert reply.buttons is None
    assert "تعذّر دمجها" in reply.text
    assert "real reason" in reply.text


def test_format_produced_with_zero_clips_and_no_stitched_path_reports_cost_without_a_video():
    conversation, _ = _conversation([])
    screenplay = DialogueScreenplay(title="T", premise="p", characters=[], lines=[])
    produced = ProducedDialogueEpisode(screenplay=screenplay, clips=[], stitched_path=None)

    reply = conversation.format_produced("chat1", produced)

    assert reply.video_path is None
    assert reply.video_paths == []
    assert reply.buttons is None


def test_format_produced_reports_lines_skipped_by_the_hard_cost_ceiling():
    # Owner's instruction, 2026-10-20: lines the ceiling stopped before
    # they were ever attempted must be visible by number, distinct from a
    # real attempt that failed (failed_lines) - the owner needs to know
    # these never spent a cent, not that they errored.
    conversation, _ = _conversation([])
    screenplay = DialogueScreenplay(title="T", premise="p", characters=[], lines=[])
    clip = ProducedDialogueClip(line=DialogueLine(index=0, character="X", line="hi"), local_path="/tmp/c0.mp4", cost_usd=0.4)
    skipped_line = DialogueLine(index=1, character="X", line="never attempted")
    produced = ProducedDialogueEpisode(
        screenplay=screenplay, clips=[clip], stitched_path="/tmp/out.mp4", skipped_lines=[skipped_line]
    )

    reply = conversation.format_produced("chat1", produced)

    assert "توقف التوليد عند الحد الأقصى" in reply.text
    assert "2" in reply.text  # 1-indexed line number, not the 0-indexed .index


def test_format_produced_reports_portrait_generation_failures_by_character_name():
    # Owner's instruction, 2026-10-20: a real, story-specific photo is
    # generated per character - if that fails for one of them, production
    # still continues with their cheap generic placeholder, but the owner
    # must be told by name which character that happened to, not left to
    # notice a mismatched face on their own.
    conversation, _ = _conversation([])
    screenplay = DialogueScreenplay(title="T", premise="p", characters=[], lines=[])
    clip = ProducedDialogueClip(line=DialogueLine(index=0, character="X", line="hi"), local_path="/tmp/c0.mp4", cost_usd=0.4)
    produced = ProducedDialogueEpisode(
        screenplay=screenplay, clips=[clip], stitched_path="/tmp/out.mp4",
        portrait_failures=[("فوزية", "fal.ai image job failed: real reason")],
    )

    reply = conversation.format_produced("chat1", produced)

    assert "فوزية" in reply.text
    assert "real reason" in reply.text


def test_format_produced_reports_failed_lines_by_number():
    # Real finding, 2026-10-19: a real line timing out on fal.ai used to
    # silently discard every clip already paid for and never told the
    # owner which real money bought nothing.
    conversation, _ = _conversation([])
    screenplay = DialogueScreenplay(title="T", premise="p", characters=[], lines=[])
    clip = ProducedDialogueClip(line=DialogueLine(index=0, character="X", line="hi"), local_path="/tmp/c0.mp4", cost_usd=0.4)
    failed_line = DialogueLine(index=1, character="X", line="bye")
    produced = ProducedDialogueEpisode(
        screenplay=screenplay, clips=[clip], stitched_path="/tmp/out.mp4",
        failed_lines=[(failed_line, "fal.ai job did not complete within 600s")],
    )

    reply = conversation.format_produced("chat1", produced)

    assert "فشلت 1" in reply.text
    assert "2" in reply.text  # 1-indexed line number shown to the owner, not the 0-indexed .index


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
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "beat one", ""), ("فوزية", "beat two", ""))
    conversation, _ = _conversation([cast, lines])
    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea")

    conversation.handle_cancel("chat1")
    reply, screenplay = conversation.handle_approve("chat1")

    assert screenplay is None
    assert "/new" in reply.text


def test_chats_are_independent():
    cast1 = _cast_reply(("فوزية", "female"))
    lines1 = _lines_reply(("فوزية", "chat1 beat", ""), ("فوزية", "chat1 beat two", ""))
    cast2 = _cast_reply(("جويدة", "female"))
    lines2 = _lines_reply(("جويدة", "chat2 beat", ""), ("جويدة", "chat2 beat two", ""))
    conversation, _ = _conversation([cast1, lines1, cast2, lines2])

    conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea one")
    conversation.handle_new("chat2", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea two")

    reply1 = conversation.handle_edit("chat1", "1 x")
    assert "chat2 beat" not in reply1.text


def test_split_for_telegram_returns_the_whole_text_as_one_chunk_when_under_the_limit():
    assert split_for_telegram("short text") == ["short text"]


def test_split_for_telegram_splits_long_text_at_a_real_newline_boundary():
    # Real gap found, 2026-10-20: the Bot API's own sendMessage rejects
    # any text over its real 4096-character limit outright rather than
    # splitting it - a long screenplay or produced-result message had no
    # chunking logic anywhere. A test that only checks the result has
    # more than one chunk would pass even if a chunk were cut mid-word -
    # it must assert no chunk exceeds the real limit AND that rejoining
    # the chunks reconstructs the original text exactly.
    line = "سطر حوار طويل بما فيه الكفاية ليتكرر ويتجاوز الحد الحقيقي لرسائل تيليجرام\n"
    text = line * 200  # well over TELEGRAM_MESSAGE_LIMIT

    chunks = split_for_telegram(text, limit=100)

    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # Rejoining with the real newline the splitter consumed between chunks
    # must reconstruct the original text exactly - no content lost or duplicated.
    assert "\n".join(chunks) == text


def test_split_for_telegram_hard_splits_when_a_single_chunk_has_no_newline():
    text = "x" * 250  # one unbroken run, no newline to break on at all

    chunks = split_for_telegram(text, limit=100)

    assert [len(c) for c in chunks] == [100, 100, 50]
    assert "".join(chunks) == text


def test_split_for_telegram_default_limit_stays_under_the_real_bot_api_cap():
    # The real Bot API cap is 4096 - TELEGRAM_MESSAGE_LIMIT must stay
    # comfortably under it (margin for Telegram's own real newline
    # counting, per real developer reports), never at or above it.
    assert TELEGRAM_MESSAGE_LIMIT < 4096
