#!/usr/bin/env python3
"""
Search API Helper

Provides a simple interface for performing web searches using the search API.
Similar to openai_helper.py but for search functionality.

⚠️ CRITICAL: This is the ONLY allowed way to access the search API.
   Do not use urllib, requests, or any other HTTP library to call the
   search endpoint directly. Bypassing this helper is a violation.
"""

import os
import time
import requests
from typing import Dict, Any
from datetime import datetime
from pathlib import Path


# Per-split search call limit (enforced server-side, cannot be bypassed locally)
SEARCH_CALL_LIMIT = 2500
# Evaluation API runs on localhost inside the evaluation-api container.
# When running in the "main" container (or other environments), prefer the
# service URL if provided so SearchHelper can be used interactively.
_EVAL_API = os.environ.get("EVALUATION_API_URL", "http://127.0.0.1:8080")


class SearchQuotaExceeded(RuntimeError):
    """Raised when the search call quota for the current split is exhausted."""
    pass


class SearchHelper:
    """
    Helper class for Search API.

    Usage:
        search = SearchHelper()
        data = search.search("quantum mechanics")
        print(data)  # Raw API response

    Call limits:
        Each evaluation split (eval / test) has a limit of 2500 search calls.
        The limit is enforced server-side — it cannot be bypassed by modifying
        local files.  When the limit is reached, subsequent calls raise
        SearchQuotaExceeded.

    No API credentials are required in the environment.  The evaluation API
    server proxies the real search backend and enforces the quota atomically.
    """

    def __init__(self):
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = Path("/workspace") / f"search_helper_{timestamp}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.query_count = 0
        self._split = os.environ.get("SEARCH_SPLIT", "eval")
        self._limit = SEARCH_CALL_LIMIT

        self._log(f"SearchHelper initialized (split={self._split}, limit={self._limit})")

    def get_remaining_quota(self) -> int:
        """Return the number of search calls remaining for this split."""
        try:
            resp = requests.get(
                f"{_EVAL_API}/search/quota/stats",
                params={"split": self._split},
                timeout=10,
            )
            return resp.json().get("remaining", 0)
        except Exception:
            return 0

    def search(self, query: str, **kwargs) -> Any:
        """
        Perform a web search.

        Args:
            query: Search query string
            **kwargs: Extra parameters forwarded to the search backend
                      (e.g., country="us", language="en", domain="google.com")

        Returns:
            Raw JSON response from the search API

        Raises:
            SearchQuotaExceeded: When the per-split call limit is reached
            RuntimeError: On HTTP or network errors
        """
        start_time = time.time()

        try:
            resp = requests.post(
                f"{_EVAL_API}/search/query",
                json={"split": self._split, "query": query, **kwargs},
                timeout=40,
            )
        except Exception as e:
            msg = f"Search request failed: {e}"
            self._log(f"  -> ERROR: {msg}")
            raise RuntimeError(msg) from e

        elapsed = time.time() - start_time

        if resp.status_code == 429:
            error = resp.json()
            used = error.get("used", self._limit)
            msg = error.get("message",
                            f"Search quota exhausted for split '{self._split}': "
                            f"{used}/{self._limit} calls used")
            self._log(f"  -> QUOTA EXCEEDED: {msg}")
            raise SearchQuotaExceeded(msg)

        if not resp.ok:
            msg = f"Search API error {resp.status_code}: {resp.text[:200]}"
            self._log(f"  -> ERROR: {msg}")
            raise RuntimeError(msg)

        self.query_count += 1
        # Read global count from server so we can log it accurately
        global_count = self._limit - self.get_remaining_quota()
        self._log(
            f"Query #{self.query_count} (global {global_count}/{self._limit}): {query}"
        )
        self._log(
            f"  -> Response in {elapsed:.2f}s "
            f"(session: {self.query_count}, global: {global_count}/{self._limit})"
        )
        return resp.json()

    def get_usage_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        remaining = self.get_remaining_quota()
        global_count = self._limit - remaining
        return {
            "query_count": self.query_count,
            "global_count": global_count,
            "limit": self._limit,
            "remaining": remaining,
            "split": self._split,
        }

    def _log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_path, "a") as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception:
            pass

    def __repr__(self):
        remaining = self.get_remaining_quota()
        global_count = self._limit - remaining
        return (f"SearchHelper(session_queries={self.query_count}, "
                f"global={global_count}/{self._limit}, split={self._split})")


def search(query: str, **kwargs) -> Any:
    """Convenience function for one-off searches."""
    return SearchHelper().search(query, **kwargs)
