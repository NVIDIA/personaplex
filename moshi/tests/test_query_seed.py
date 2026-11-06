"""Regression tests for the optional ``seed`` query parameter of /api/chat.

``ServerState.handle_chat`` reads the seed via ``_query_seed``. It previously
used ``int(request["seed"]) if "seed" in request.query else None``, but
``BaseRequest.__getitem__`` only looks up path (match_info) parameters, so any
client passing ``?seed=`` in the query string crashed the handler with
``KeyError`` (HTTP 500) instead of getting seeded inference.
"""

from aiohttp import test_utils

from moshi.utils.connection import query_seed


def test_seed_present_in_query():
    request = test_utils.make_mocked_request("GET", "/api/chat?seed=1234")
    assert query_seed(request) == 1234


def test_seed_absent_returns_none():
    request = test_utils.make_mocked_request("GET", "/api/chat")
    assert query_seed(request) is None


def test_seed_minus_one_is_preserved():
    # -1 is the documented "disable seeding" value; parsing must not drop it.
    request = test_utils.make_mocked_request("GET", "/api/chat?seed=-1")
    assert query_seed(request) == -1
