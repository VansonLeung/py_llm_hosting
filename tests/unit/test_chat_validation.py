from src.api.chat import ChatMessage, _message_has_vision_content


def test_text_only_list_is_not_vision():
    message = ChatMessage(role="user", content=[{"type": "text", "text": "hi"}])
    assert _message_has_vision_content(message) is False


def test_image_block_triggers_vision():
    message = ChatMessage(
        role="user",
        content=[
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ],
    )
    assert _message_has_vision_content(message) is True
