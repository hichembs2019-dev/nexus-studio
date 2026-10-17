from src.integrations.telegram_client import MockTelegramClient


def test_send_message_records_text_and_buttons_and_returns_a_message_id():
    client = MockTelegramClient()

    message_id = client.send_message("chat1", "hello", buttons=[("A", "a"), ("B", "b")])

    assert client.sent_messages == [{"chat_id": "chat1", "text": "hello", "buttons": [("A", "a"), ("B", "b")]}]
    assert message_id


def test_send_video_records_the_path_and_caption():
    client = MockTelegramClient()

    client.send_video("chat1", "/tmp/out.mp4", "caption")

    assert client.sent_videos == [{"chat_id": "chat1", "video_path": "/tmp/out.mp4", "caption": "caption"}]


def test_answer_callback_query_records_the_id():
    client = MockTelegramClient()

    client.answer_callback_query("cbq-1")

    assert client.answered_callbacks == ["cbq-1"]


def test_each_send_gets_a_distinct_message_id():
    client = MockTelegramClient()

    id1 = client.send_message("chat1", "a")
    id2 = client.send_video("chat1", "/tmp/x.mp4", "c")

    assert id1 != id2
