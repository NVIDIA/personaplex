"""Regression tests for an empty (None) text prompt.

``LMGen.text_prompt_tokens`` is Optional and defaults to None; both the server
and the offline entrypoint set it to None when the client supplies an empty
text prompt. ``_step_text_prompt_core`` iterated the value unconditionally, so
a None prompt crashed the prompt phase with
``TypeError: 'NoneType' object is not iterable``.
"""

from moshi.models.lm import LMGen


def _bare_lm_gen() -> LMGen:
    # Bypass __init__, which requires a real LMModel. The text-prompt core
    # only touches the attributes set below.
    return object.__new__(LMGen)


def test_none_text_prompt_is_skipped():
    gen = _bare_lm_gen()
    gen.text_prompt_tokens = None
    # Previously raised TypeError: 'NoneType' object is not iterable.
    assert list(gen._step_text_prompt_core()) == []


def test_text_prompt_steps_each_token_in_order():
    gen = _bare_lm_gen()
    gen.text_prompt_tokens = [10, 20, 30]
    stepped = []
    gen.step = lambda **kwargs: stepped.append(kwargs["text_token"])
    gen._encode_zero_frame = lambda: None
    gen._encode_sine_frame = lambda: None
    list(gen._step_text_prompt_core())
    assert stepped == [10, 20, 30]
