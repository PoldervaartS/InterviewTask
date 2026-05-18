"""Tests for fetch_text (fetcher.py)."""
from unittest.mock import MagicMock, patch

import pytest

from quiz_agent.fetcher import fetch_text


def _mock_response(status_code: int = 200, json_data: dict | None = None, text: str = "", content_type: str = "text/html"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.headers = {"content-type": content_type}
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


class TestFetchTextWikipedia:
    def test_wikipedia_uses_query_api(self):
        """Wikipedia URLs must go through action=query API, not raw HTML fetch."""
        api_resp = _mock_response(
            json_data={
                "query": {
                    "pages": {"12345": {"extract": "Austin history text here."}}
                }
            },
            content_type="application/json",
        )
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = api_resp
            result = fetch_text("https://en.wikipedia.org/w/index.php?title=History_of_Austin,_Texas")

        call_args = MockClient.return_value.__enter__.return_value.get.call_args
        called_url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert "api.php" in called_url, "Wikipedia fetches must use the action=query API endpoint"
        assert result == "Austin history text here."

    def test_wikipedia_extract_truncated_to_max_chars(self):
        long_text = "x" * 50_000
        api_resp = _mock_response(
            json_data={
                "query": {"pages": {"1": {"extract": long_text}}}
            },
            content_type="application/json",
        )
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = api_resp
            result = fetch_text("https://en.wikipedia.org/wiki/Some_Article")

        assert len(result) <= 40_000

    def test_wikipedia_user_agent_includes_contact(self):
        """User-Agent must include contact info per Wikimedia bot policy."""
        api_resp = _mock_response(
            json_data={"query": {"pages": {"1": {"extract": "text"}}}},
            content_type="application/json",
        )
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = api_resp
            fetch_text("https://en.wikipedia.org/wiki/Some_Article")

        call_kwargs = MockClient.return_value.__enter__.return_value.get.call_args[1]
        ua = call_kwargs.get("headers", {}).get("User-Agent", "")
        assert "@" in ua, "User-Agent must contain contact email per Wikimedia policy"


class TestFetchTextGeneric:
    def test_non_wikipedia_url_uses_direct_get(self):
        """Non-Wikipedia URLs should be fetched directly."""
        html_resp = _mock_response(text="<p>Hello world</p>", content_type="text/html")
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = html_resp
            result = fetch_text("https://example.com/page")

        call_args = MockClient.return_value.__enter__.return_value.get.call_args
        called_url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert called_url == "https://example.com/page"
        assert "Hello world" in result

    def test_http_error_propagates(self):
        resp = _mock_response(status_code=404)
        resp.raise_for_status.side_effect = Exception("404 Not Found")
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = resp
            with pytest.raises(Exception, match="404"):
                fetch_text("https://example.com/missing")
