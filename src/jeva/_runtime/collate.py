from __future__ import annotations


def state_question_text(state: str, question: str, decision_type: str) -> str:
    return f"Decision type: {decision_type}\nQuestion: {question}\nState:\n{state}"


def option_text(option: str, decision_type: str) -> str:
    return f"Option for {decision_type}: {option}"


def tokenize_texts(tokenizer, texts, max_length, policy, *, field):
    if max_length < 1:
        raise ValueError("maximum token lengths must be positive")
    if policy not in {"error", "truncate"}:
        raise ValueError("truncation must be 'error' or 'truncate'")
    encoded = tokenizer(texts, padding=False, truncation=False)
    lengths = [len(row) for row in encoded["input_ids"]]
    overflow = [(i, n) for i, n in enumerate(lengths) if n > max_length]
    if overflow and policy == "error":
        index, length = overflow[0]
        raise ValueError(
            f"{field}[{index}] has {length} tokens; limit is {max_length}. "
            "Shorten the input, increase the configured limit, or explicitly "
            "set truncation='truncate'."
        )
    truncated = sum(max(0, n - max_length) for n in lengths)
    if overflow:
        encoded = tokenizer(texts, padding=False, truncation=True, max_length=max_length)
    rows = [{key: value[i] for key, value in encoded.items()} for i in range(len(texts))]
    return rows, [min(n, max_length) for n in lengths], truncated
