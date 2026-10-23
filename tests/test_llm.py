def test_answer_budget_shrinks_to_the_context_window():
    from codeladder.llm import _fitting_max_tokens

    class E(Exception):
        status_code = 400

    msg = (
        "Error code: 400 - {'object': 'error', 'message': \"This model's maximum context length "
        "is 8192 tokens. However, you requested 9674 tokens (7626 in the messages, 2048 in the "
        'completion). Please reduce the length of the messages or completion."}'
    )
    assert _fitting_max_tokens(E(msg), 2048) == 8192 - 7626 - 16
    assert _fitting_max_tokens(E(msg.replace("7626", "8000")), 2048) is None  # too little room
    assert _fitting_max_tokens(E("rate limited"), 2048) is None
