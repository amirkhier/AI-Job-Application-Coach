"""
Unit tests for the Job Search Agent.

Covers:
- search_jobs() with location geocoding and LLM generation
- get_city_center() geocoding
- find_nearby_offices() Overpass API
- _haversine() distance calculation
- Job listing normalisation and fallback
- JSON parsing
- Error handling for external API failures
- Performance timing
"""

import json
import math
import time
import requests
import pytest
from unittest.mock import patch, MagicMock

from app.agents.job_search import JobSearchAgent


# ================================================================ #
#  Fixtures
# ================================================================ #

@pytest.fixture(scope="module")
def agent():
    return JobSearchAgent()


# ================================================================ #
#  Geocoding — get_city_center
# ================================================================ #

class TestGetCityCenter:

    def test_known_city(self, agent):
        """Tel Aviv should geocode successfully."""
        result = agent.get_city_center("Tel Aviv")

        assert result["found"] is True
        assert result["lat"] is not None
        assert result["lon"] is not None
        assert result["city"] == "Tel Aviv"
        # Rough sanity check for Tel Aviv coords
        assert 31.0 < result["lat"] < 33.0
        assert 34.0 < result["lon"] < 36.0

    def test_another_city(self, agent):
        """Berlin should geocode successfully."""
        result = agent.get_city_center("Berlin")
        assert result["found"] is True
        assert 52.0 < result["lat"] < 53.0

    def test_nonexistent_city(self, agent):
        """Made-up city should return found=False."""
        result = agent.get_city_center("Xyznonexistentcity12345")
        assert result["found"] is False
        assert result["lat"] is None

    def test_geocode_timeout(self, agent):
        """Network failure should return found=False, not crash."""
        with patch("app.agents.job_search.requests.get", side_effect=requests.RequestException("timeout")):
            result = agent.get_city_center("London")
        assert result["found"] is False


# ================================================================ #
#  Overpass — find_nearby_offices
# ================================================================ #

class TestFindNearbyOffices:

    def test_returns_list(self, agent):
        """Should return a list (possibly empty) of offices."""
        # Use Tel Aviv coords
        result = agent.find_nearby_offices(32.0853, 34.7818, radius=3000)
        assert isinstance(result, list)

    def test_each_office_has_keys(self, agent):
        """Each office dict should have name, type, lat, lon, distance_m."""
        results = agent.find_nearby_offices(32.0853, 34.7818, radius=5000)
        for office in results[:3]:  # check first few
            assert "name" in office
            assert "lat" in office
            assert "lon" in office
            assert "distance_m" in office

    def test_sorted_by_distance(self, agent):
        """Results should be sorted by distance ascending."""
        results = agent.find_nearby_offices(32.0853, 34.7818, radius=5000)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i]["distance_m"] <= results[i + 1]["distance_m"]

    def test_overpass_failure_returns_empty(self, agent):
        """If Overpass API fails, return empty list."""
        with patch("app.agents.job_search.requests.post", side_effect=requests.RequestException("API down")):
            result = agent.find_nearby_offices(52.52, 13.405)
        assert result == []


# ================================================================ #
#  Haversine distance
# ================================================================ #

class TestHaversine:

    def test_same_point_is_zero(self):
        assert JobSearchAgent._haversine(32.0, 34.0, 32.0, 34.0) == 0.0

    def test_known_distance(self):
        """Tel Aviv to Jerusalem is ~55-65 km."""
        dist = JobSearchAgent._haversine(32.0853, 34.7818, 31.7683, 35.2137)
        assert 50_000 < dist < 70_000, f"Distance {dist:.0f}m not in expected range"

    def test_symmetry(self):
        d1 = JobSearchAgent._haversine(32.0, 34.0, 33.0, 35.0)
        d2 = JobSearchAgent._haversine(33.0, 35.0, 32.0, 34.0)
        assert abs(d1 - d2) < 0.01


# ================================================================ #
#  Full job search
# ================================================================ #

class TestSearchJobs:

    def test_search_returns_jobs(self, agent):
        """Full search should return a list of job listings."""
        result = agent.search_jobs(
            query="Python Developer",
            location="Tel Aviv",
            count=3,
        )
        assert "jobs" in result
        assert "total_found" in result
        assert "location_info" in result
        assert "processing_time" in result
        assert len(result["jobs"]) > 0

    def test_job_listing_structure(self, agent):
        """Each job listing should have the expected keys."""
        result = agent.search_jobs("Backend Engineer", "Berlin", count=2)

        for job in result["jobs"]:
            assert "title" in job
            assert "company" in job
            assert "location" in job
            assert "description" in job
            assert "match_score" in job
            assert "key_skills" in job

    def test_count_respected(self, agent):
        """Should generate approximately the requested number of jobs."""
        result = agent.search_jobs("Data Scientist", "London", count=2)
        assert len(result["jobs"]) <= 5  # Allow some flexibility
        assert len(result["jobs"]) >= 1

    def test_performance_full_search(self, agent):
        """Full search pipeline should complete within 90 seconds."""
        start = time.time()
        agent.search_jobs("Engineer", "New York", count=2)
        elapsed = time.time() - start
        assert elapsed < 90, f"Full search took {elapsed:.1f}s"


# ================================================================ #
#  Normalisation and fallback
# ================================================================ #

class TestNormalisationAndFallback:

    def test_normalize_job_listings(self, agent):
        """Normalise should fill missing keys with defaults."""
        raw = [{"title": "Dev"}, {"not_a_real_key": True}]
        result = agent._normalize_job_listings(raw)
        assert all("title" in j for j in result)
        assert all("company" in j for j in result)

    def test_fallback_listings(self, agent):
        """Fallback should return valid job dicts."""
        fallback = agent._fallback_listings("Engineer", "Berlin", "mid", True)
        assert len(fallback) >= 1
        assert all("title" in j for j in fallback)

    def test_normalize_skips_non_dicts(self, agent):
        raw = [{"title": "Good"}, "bad_entry", 42, None]
        result = agent._normalize_job_listings(raw)
        assert len(result) == 1  # Only the dict passes


# ================================================================ #
#  JSON parsing
# ================================================================ #

class TestJSONParsing:

    def test_parse_array(self):
        raw = '[{"title": "Dev"}]'
        result = JobSearchAgent._parse_llm_json(raw)
        assert isinstance(result, list)

    def test_parse_object(self):
        raw = '{"jobs": [{"title": "Dev"}]}'
        result = JobSearchAgent._parse_llm_json(raw)
        assert "jobs" in result

    def test_parse_with_fences(self):
        raw = '```json\n[{"title": "Dev"}]\n```'
        result = JobSearchAgent._parse_llm_json(raw)
        assert isinstance(result, list)

    def test_unparseable_returns_none(self):
        result = JobSearchAgent._parse_llm_json("total garbage")
        assert result is None
