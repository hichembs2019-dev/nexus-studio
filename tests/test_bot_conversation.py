from __future__ import annotations

import json

from src.agents.dialogue_writer import DialogueWriterAgent
from src.agents.publisher import PublishingAgent
from src.bot.conversation import ESTIMATED_SECONDS_PER_LINE, BotConversation, TELEGRAM_MESSAGE_LIMIT, split_for_telegram
from src.integrations.fal_video_client import PRICE_PER_SECOND_USD
from src.integrations.image_client import PRICE_PER_MEGAPIXEL_USD
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


def test_new_cost_estimate_includes_real_portrait_generation_cost():
    # Real bug found by code review, 2026-10-21: the estimate shown
    # before Approve only ever counted spoken-line video cost - never
    # the real per-character portrait generation DialoguePipeline also
    # spends (fal-ai/flux/dev, $0.025/megapixel), once per character who
    # appears in any line at all. The one number this owner checks
    # before committing real money silently left out a real cost
    # component every single time.
    cast = _cast_reply(("فوزية", "female"), ("جويدة", "female"))
    lines = _lines_reply(("فوزية", "beat one", ""), ("جويدة", "beat two", ""))
    conversation, _ = _conversation([cast, lines])

    reply = conversation.handle_new("chat1", f"{int(2 * ESTIMATED_SECONDS_PER_LINE)} idea")

    from src.integrations.image_client import PRICE_PER_MEGAPIXEL_USD

    expected_cost = 2 * ESTIMATED_SECONDS_PER_LINE * PRICE_PER_SECOND_USD["480p"] + 2 * PRICE_PER_MEGAPIXEL_USD
    assert f"${expected_cost:.2f}" in reply.text


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
    # to reflect just that one line, not both. Both characters still
    # appear in the screenplay (one speaking, one captioned), so both
    # still need a real generated portrait - that real cost must still
    # be in the estimate too (real bug found by code review, 2026-10-21:
    # it never was, before this).
    expected_cost = 1 * ESTIMATED_SECONDS_PER_LINE * PRICE_PER_SECOND_USD["480p"] + 2 * PRICE_PER_MEGAPIXEL_USD
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
    # Owner's standing instruction, 2026-10-21 ("احذف الستة صور"): a real,
    # story-specific photo is generated per character - if that fails,
    # there is no fallback any more, so that character's lines fail
    # (zero cost) instead of production continuing with a placeholder
    # face. Real bug found by code review the same day: this message
    # used to say "(a placeholder photo was used instead)" - false once
    # the fallback itself was deleted, and left unfixed when it was -
    # the owner must be told the truth (the line failed), not a claim
    # about a placeholder that no longer exists anywhere in this code.
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
    assert "استُخدمت صورة افتراضية" not in reply.text  # no fallback exists any more - never claim one was used
    assert "فلم تُنتَج أسطرها" in reply.text  # must say the lines failed instead


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


# Owner's standing instruction, 2026-10-21 ("اريد الحديث معه كما اتحدث
# معك ... محادثة حرة كاملة مع نقاش وتعديل جماعي للقصة") - handle_message() tests.


def _router_reply(intent, reply="رد", edits=None, caption_lines=None):
    return json.dumps({"intent": intent, "reply": reply, "edits": edits or [], "caption_lines": caption_lines or []})


def test_handle_message_accumulate_stores_the_text_and_replies_conversationally():
    router = _router_reply("accumulate", reply="تم، أكمل.")
    conversation, _ = _conversation([router])

    reply = conversation.handle_message("chat1", "كان يا ما كان رجل وحيد")

    assert reply.text == "تم، أكمل."
    assert reply.buttons is None


def test_handle_message_discuss_never_triggers_real_production():
    # Real safety line that must never move: production (DialoguePipeline.
    # produce(), the one real-money call) can ONLY start from
    # handle_approve(), triggered by the explicit ✅ button press - never
    # from a free message, however it is phrased. This asserts the whole
    # exchange ends with conversation.handle_approve's own precondition
    # still unmet unless a real screenplay was separately built and
    # approved - "وافقت انتج الآن" alone must never produce one.
    router = _router_reply("discuss", reply="اضغط زر ✅ موافقة لبدء الإنتاج الحقيقي.")
    conversation, _ = _conversation([router])

    reply = conversation.handle_message("chat1", "وافقت، انتج الفيديو الآن")

    assert "✅" in reply.text
    assert reply.buttons is None
    # No screenplay was ever built, so approval is still structurally impossible.
    approve_reply, screenplay = conversation.handle_approve("chat1")
    assert screenplay is None
    assert "/new" in approve_reply.text


def test_handle_message_build_screenplay_uses_verbatim_mode_when_real_quotes_exist():
    router = _router_reply("build_screenplay", reply="جارٍ بناء السيناريو من كلامك الحقيقي.")
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "مرحبا", "smiling"))
    conversation, _ = _conversation([router, cast, lines])

    reply = conversation.handle_message("chat1", 'فوزية قالت: «مرحبا»')

    assert "جارٍ بناء السيناريو" in reply.text
    assert "مرحبا" in reply.text
    assert reply.buttons == [("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")]


def test_handle_message_build_screenplay_falls_back_to_estimated_mode_with_no_quotes():
    router = _router_reply("build_screenplay", reply="جارٍ بناء سيناريو من فكرتك.")
    cast = _cast_reply(("فوزية", "female"))
    line_count = max(2, round(60.0 / ESTIMATED_SECONDS_PER_LINE))
    lines = _lines_reply(*[("فوزية", f"سطر {i}", "") for i in range(line_count)])
    conversation, _ = _conversation([router, cast, lines])

    reply = conversation.handle_message("chat1", "فكرة بلا أي كلام منقول حرفيًا عن امرأة وحيدة")

    assert reply.buttons == [("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")]


def test_handle_message_build_screenplay_uses_previously_accumulated_text_too():
    router_accumulate = _router_reply("accumulate", reply="تم.")
    router_build = _router_reply("build_screenplay", reply="جارٍ البناء.")
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "مرحبا", "smiling"))
    conversation, _ = _conversation([router_accumulate, router_build, cast, lines])

    conversation.handle_message("chat1", 'فوزية قالت: «مرحبا»')
    reply = conversation.handle_message("chat1", "جاهز، ابن السيناريو الآن")

    assert "مرحبا" in reply.text


def test_handle_message_never_auto_builds_a_screenplay_from_the_debounce_path():
    # Real, live bug found 2026-10-21, directly from the owner's own
    # confusion ("لم اعد افهمك... لم اعد افهم البوت"): in the real
    # deployed bot, handle_message() is ONLY ever called with an empty
    # text from the debounce timer - runner.py's _text handler never
    # calls it with real text directly. If "build_screenplay" could fire
    # from that empty-text (pulled_from_pending) path, the router would
    # silently auto-build and CONSUME the owner's pending story text
    # about 1.5s after he went quiet - racing his own following /full or
    # /new command, which would then find nothing left to build from.
    # This proves the debounce path never auto-builds, and the real
    # story text is preserved (not lost) for the owner's own explicit
    # /full right after.
    router_build = _router_reply("build_screenplay", reply="يبدو جاهزًا للبناء.")
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "مرحبا", "smiling"))
    conversation, _ = _conversation([router_build, cast, lines])
    conversation.handle_plain_text("chat1", 'فوزية قالت: «مرحبا»')

    reply = conversation.handle_message("chat1", "")  # debounce-driven call

    assert "/full" in reply.text
    assert "/new" in reply.text
    # No screenplay was built - handle_approve still has nothing to approve.
    approve_reply, screenplay = conversation.handle_approve("chat1")
    assert screenplay is None
    # The real story text must have survived, ready for /full right now -
    # not silently consumed by the auto-build attempt above.
    full_reply = conversation.handle_full("chat1", "")
    assert "مرحبا" in full_reply.text


def test_handle_message_edit_lines_applies_the_owners_exact_quoted_words():
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "مرحبا", "smiling"))
    router_build = _router_reply("build_screenplay", reply="بُني.")
    router_edit = _router_reply("edit_lines", reply="عدّلت السطر الأول.", edits=[{"line": 1, "text": "لن أسامحك"}])
    conversation, _ = _conversation([router_build, cast, lines, router_edit])

    conversation.handle_message("chat1", 'فوزية قالت: «مرحبا»')
    reply = conversation.handle_message("chat1", 'اجعل السطر الأول يقول حرفيًا: «لن أسامحك»')

    assert "لن أسامحك" in reply.text
    assert reply.buttons == [("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")]


def test_handle_message_set_captions_marks_the_real_line_as_free():
    cast = _cast_reply(("فوزية", "female"), ("جويدة", "female"))
    lines = _lines_reply(("فوزية", "واحد", ""), ("جويدة", "اثنان", ""))
    router_build = _router_reply("build_screenplay", reply="بُني.")
    router_caption = _router_reply("set_captions", reply="حوّلت سطر جويدة لنص على الشاشة.", caption_lines=[2])
    conversation, _ = _conversation([router_build, cast, lines, router_caption])

    conversation.handle_message("chat1", 'فوزية: «واحد» جويدة: «اثنان»')
    reply = conversation.handle_message("chat1", "اجعل سطر جويدة نصًا مكتوبًا لا صوتًا")

    assert "نص على الشاشة" in reply.text


def test_handle_message_edit_lines_without_a_draft_is_structurally_refused():
    # route_message() itself already raises for this (a real code-level
    # gate), but this proves handle_message surfaces that as a clear
    # reply rather than an uncaught exception reaching the owner.
    router = _router_reply("edit_lines", reply="x", edits=[{"line": 1, "text": "y"}])
    conversation, _ = _conversation([router])

    reply = conversation.handle_message("chat1", "غيّر السطر الأول")

    assert "تعذّر فهم رسالتك" in reply.text


def test_handle_message_with_empty_text_routes_the_accumulated_pending_text_instead():
    # Real risk found by code review, 2026-10-21: Telegram's own client
    # auto-splits one long typed/pasted message into several separate
    # text updates - runner.py debounces these and calls handle_message
    # with an empty string once the owner has gone quiet, so the FULL
    # coalesced story (not a mid-sentence fragment) must be what actually
    # gets routed through the LLM.
    router = _router_reply("accumulate", reply="تم.")
    reasoning = MockAgentReasoningClient(scripted_replies=[router])
    conversation = BotConversation(DialogueWriterAgent(reasoning), PublishingAgent({}))
    conversation.handle_plain_text("chat1", "جزء أول من القصة")
    conversation.handle_plain_text("chat1", "جزء ثانٍ من نفس القصة")

    reply = conversation.handle_message("chat1", "")

    assert reply.text == "تم."
    call_user_prompt = reasoning.calls[0][1]
    assert "جزء أول من القصة" in call_user_prompt
    assert "جزء ثانٍ من نفس القصة" in call_user_prompt


def test_handle_message_with_empty_text_and_nothing_pending_does_nothing():
    conversation, _ = _conversation([])

    reply = conversation.handle_message("chat1", "")

    assert reply.text == ""


# Real incident, 2026-10-21, found live the first time the feature was
# actually used: pending_text was cleared unconditionally once pulled by
# the debounce path, but only ever restored by the "accumulate" and
# "build_screenplay" branches - "discuss"/"edit_lines"/"set_captions"
# silently discarded the owner's real accumulated story text forever.
# Each test below proves the text SURVIVES (a second debounce-pulled call
# still has real text to route, rather than returning empty immediately)
# and is not duplicated (the router sees it exactly once per call).


def test_handle_message_discuss_preserves_pending_text_pulled_from_debounce():
    router_discuss = _router_reply("discuss", reply="سؤال جيد.")
    router_accumulate = _router_reply("accumulate", reply="تم.")
    reasoning = MockAgentReasoningClient(scripted_replies=[router_discuss, router_accumulate])
    conversation = BotConversation(DialogueWriterAgent(reasoning), PublishingAgent({}))
    conversation.handle_plain_text("chat1", "جزء من قصة حقيقية لم تتحول لسيناريو بعد")

    first = conversation.handle_message("chat1", "")
    assert first.text == "سؤال جيد."

    second = conversation.handle_message("chat1", "")
    assert second.text == "تم."
    assert reasoning.calls[0][1].count("جزء من قصة حقيقية لم تتحول لسيناريو بعد") == 1
    assert reasoning.calls[1][1].count("جزء من قصة حقيقية لم تتحول لسيناريو بعد") == 1


def test_handle_message_edit_lines_preserves_unrelated_pending_text_pulled_from_debounce():
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "مرحبا", "smiling"))
    router_build = _router_reply("build_screenplay", reply="بُني.")
    router_edit = _router_reply("edit_lines", reply="عدّلت.", edits=[{"line": 1, "text": "لن أسامحك"}])
    router_accumulate = _router_reply("accumulate", reply="تم.")
    reasoning = MockAgentReasoningClient(scripted_replies=[router_build, cast, lines, router_edit, router_accumulate])
    conversation = BotConversation(DialogueWriterAgent(reasoning), PublishingAgent({}))

    conversation.handle_message("chat1", 'فوزية قالت: «مرحبا»')
    conversation.handle_plain_text("chat1", "مادة قصة أخرى منفصلة لم تُستخدم بعد")

    edit_reply = conversation.handle_message("chat1", "")
    assert "لن أسامحك" in edit_reply.text

    again = conversation.handle_message("chat1", "")
    assert again.text == "تم."


def test_handle_message_set_captions_preserves_unrelated_pending_text_pulled_from_debounce():
    cast = _cast_reply(("فوزية", "female"))
    lines = _lines_reply(("فوزية", "مرحبا", "smiling"))
    router_build = _router_reply("build_screenplay", reply="بُني.")
    router_caption = _router_reply("set_captions", reply="حوّلت.", caption_lines=[1])
    router_accumulate = _router_reply("accumulate", reply="تم.")
    reasoning = MockAgentReasoningClient(scripted_replies=[router_build, cast, lines, router_caption, router_accumulate])
    conversation = BotConversation(DialogueWriterAgent(reasoning), PublishingAgent({}))

    conversation.handle_message("chat1", 'فوزية قالت: «مرحبا»')
    conversation.handle_plain_text("chat1", "مادة قصة أخرى منفصلة لم تُستخدم بعد")

    caption_reply = conversation.handle_message("chat1", "")
    assert "نص على الشاشة" in caption_reply.text

    again = conversation.handle_message("chat1", "")
    assert again.text == "تم."
