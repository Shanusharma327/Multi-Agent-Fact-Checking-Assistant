"""Tests for URL canonicalisation, domain extraction and the leakage filter."""

from __future__ import annotations

import pytest

from fact_checker.urls import (
    canonical_set,
    canonical_url,
    domain_set,
    filter_leakage,
    registered_domain,
    same_document,
    unwrap_archive,
)


class TestUnwrapArchive:
    def test_wayback_wrapper_yields_the_target(self):
        wrapped = "https://web.archive.org/web/20201101120000/https://bbc.co.uk/news/x"
        assert unwrap_archive(wrapped) == "https://bbc.co.uk/news/x"

    def test_archive_today_wrapper(self):
        wrapped = "https://archive.ph/abc12/https://example.com/story"
        assert unwrap_archive(wrapped) == "https://example.com/story"

    def test_a_plain_url_is_returned_unchanged(self):
        assert unwrap_archive("https://example.com/a") == "https://example.com/a"

    def test_nested_wrappers_are_unwrapped(self):
        inner = "https://example.com/a"
        once = f"https://web.archive.org/web/2020/{inner}"
        twice = f"https://web.archive.org/web/2021/{once}"
        assert unwrap_archive(twice) == inner

    def test_empty_input_is_safe(self):
        assert unwrap_archive("") == ""


class TestCanonicalUrl:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("http://www.Example.com/a/b/", "https://example.com/a/b"),
            ("https://example.com", "https://example.com/"),
            ("https://example.com/a#section", "https://example.com/a"),
            ("https://example.com/a?utm_source=tw", "https://example.com/a"),
            ("https://example.com/a?id=5&fbclid=z", "https://example.com/a?id=5"),
            ("https://example.com:443/a", "https://example.com/a"),
            ("example.org/path", "https://example.org/path"),
        ],
    )
    def test_normalisation(self, raw, expected):
        assert canonical_url(raw) == expected

    def test_archive_wrapper_is_unwrapped_then_normalised(self):
        wrapped = (
            "https://web.archive.org/web/20201101/"
            "https://www.BBC.co.uk/news/article-1?utm_medium=x#top"
        )
        assert canonical_url(wrapped) == "https://bbc.co.uk/news/article-1"

    @pytest.mark.parametrize("raw", ["", "   ", "not a url", "https://localhost/x"])
    def test_unusable_input_yields_empty(self, raw):
        """An unusable URL must never match: it is absence, not a hit."""
        assert canonical_url(raw) == ""

    def test_http_and_https_collapse(self):
        assert canonical_url("http://a.com/x") == canonical_url("https://a.com/x")

    def test_query_parameter_order_is_preserved_not_sorted(self):
        """Reordering would risk changing which document a URL selects."""
        assert canonical_url("https://a.com/x?b=2&a=1") == "https://a.com/x?b=2&a=1"


class TestRegisteredDomain:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("https://news.bbc.co.uk/story", "bbc.co.uk"),
            ("https://www.example.com/a", "example.com"),
            ("https://a.b.c.example.org/x", "example.org"),
            ("https://site.com.au/x", "site.com.au"),
            ("http://192.168.1.1/a", "192.168.1.1"),
            ("", ""),
        ],
    )
    def test_extraction(self, raw, expected):
        assert registered_domain(raw) == expected


class TestSetHelpers:
    def test_canonical_set_deduplicates_equivalent_urls(self):
        urls = [
            "http://www.a.com/x",
            "https://a.com/x/",
            "https://a.com/x?utm_source=q",
            "",
        ]
        assert canonical_set(urls) == {"https://a.com/x"}

    def test_domain_set_collapses_pages_on_one_host(self):
        assert domain_set(["https://a.com/1", "https://a.com/2"]) == {"a.com"}

    def test_same_document_ignores_cosmetic_difference(self):
        assert same_document("http://WWW.A.com/x", "https://a.com/x/")

    def test_same_document_is_false_for_blank(self):
        assert not same_document("", "")


class TestLeakageFilter:
    def test_the_originating_fact_check_is_flagged(self):
        flags = filter_leakage(
            ["https://www.politifact.com/x", "https://bbc.com/y"],
            "http://politifact.com/x",
        )
        assert flags == [True, False]

    def test_an_archived_copy_of_the_fact_check_is_also_flagged(self):
        flags = filter_leakage(
            ["https://web.archive.org/web/2020/https://politifact.com/x"],
            "https://politifact.com/x",
        )
        assert flags == [True]

    def test_nothing_is_flagged_without_a_target(self):
        assert filter_leakage(["https://a.com/1", "https://b.com/2"], "") == [
            False,
            False,
        ]

    def test_blocked_domains_catch_sibling_pages(self):
        """The strict variant excludes the whole domain, not just one page."""
        flags = filter_leakage(
            ["https://politifact.com/other", "https://bbc.com/y"],
            "",
            blocked_domains=["politifact.com"],
        )
        assert flags == [True, False]
