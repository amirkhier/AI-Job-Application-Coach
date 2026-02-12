"""
Job Search Agent — Location-aware job search with geolocation services.

Combines OpenStreetMap Nominatim geocoding, Overpass API for nearby
offices/companies, and LLM-powered job matching to provide relevant,
location-enriched job search results.

Usage::

    from app.agents.job_search import JobSearchAgent

    agent = JobSearchAgent()

    # Full job search
    results = agent.search_jobs(
        query="Python backend engineer",
        location="Tel Aviv",
        experience_level="senior",
        remote_ok=True,
    )

    # Geolocation helpers
    coords = agent.get_city_center("Berlin")
    offices = agent.find_nearby_offices(52.52, 13.405, radius=5000)
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Polite User-Agent as required by Nominatim usage policy
_USER_AGENT = "AIJobApplicationCoach/1.0 (career-coaching-project)"

# Timeout for external API calls (seconds)
_HTTP_TIMEOUT = 15

# ------------------------------------------------------------------ #
#  Agent
# ------------------------------------------------------------------ #


class JobSearchAgent:
    """Location-aware job search and company discovery agent."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
    ):
        self.llm = ChatOpenAI(model=model, temperature=temperature)

        # ---- job generation prompt -----------------------------------
        self.job_generation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an expert job market analyst for the career coaching system. "
                        "Generate realistic, helpful job listings that match the user's search criteria.\n\n"
                        "RULES:\n"
                        "1. Create varied listings across different company sizes and types.\n"
                        "2. Include realistic salary ranges based on role, level, and location.\n"
                        "3. Write genuine-sounding descriptions with concrete requirements.\n"
                        "4. For Israeli locations use ILS salaries; for US locations use USD.\n"
                        "5. Mix remote/hybrid/onsite based on `remote_ok` preference.\n"
                        "6. Include companies that realistically hire for this role in this location.\n"
                        "7. If nearby companies were found via geolocation, incorporate them.\n\n"
                        "Return a JSON array of job objects with this schema:\n"
                        "[\n"
                        "  {{\n"
                        '    "title": "<job title>",\n'
                        '    "company": "<company name>",\n'
                        '    "location": "<city, region or Remote>",\n'
                        '    "description": "<2-3 sentence description with key requirements>",\n'
                        '    "url": "<plausible job board URL>",\n'
                        '    "salary_range": "<salary range string>",\n'
                        '    "remote_friendly": <true|false>,\n'
                        '    "match_score": <0.0-1.0 relevance to query>,\n'
                        '    "experience_level": "<junior|mid|senior|lead>",\n'
                        '    "key_skills": ["skill1", "skill2", "skill3"]\n'
                        "  }}\n"
                        "]\n\n"
                        "Return ONLY a valid JSON array — no markdown fences, no extra text."
                    ),
                ),
                (
                    "human",
                    (
                        "Search criteria:\n"
                        "- Query: {query}\n"
                        "- Location: {location}\n"
                        "- Experience level: {experience_level}\n"
                        "- Remote OK: {remote_ok}\n\n"
                        "Nearby companies/offices found: {nearby_companies}\n\n"
                        "Location info: {location_info}\n\n"
                        "Generate {count} relevant job listings."
                    ),
                ),
            ]
        )

        # ---- job matching / relevance prompt -------------------------
        self.matching_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a career advisor. Given a user's profile and a list of "
                        "job listings, score each job's match quality and provide brief advice.\n\n"
                        "Return JSON:\n"
                        "{{\n"
                        '  "matched_jobs": [\n'
                        "    {{\n"
                        '      "title": "<job title>",\n'
                        '      "company": "<company>",\n'
                        '      "match_score": <0.0-1.0>,\n'
                        '      "match_reasons": ["reason1", "reason2"],\n'
                        '      "gaps": ["gap1", "gap2"],\n'
                        '      "application_tip": "<one-sentence advice>"\n'
                        "    }}\n"
                        "  ],\n"
                        '  "overall_advice": "<general job search advice>",\n'
                        '  "market_insight": "<brief market observation for this search>"\n'
                        "}}\n\n"
                        "Return ONLY valid JSON."
                    ),
                ),
                (
                    "human",
                    (
                        "User profile:\n{user_profile}\n\n"
                        "Job listings:\n{job_listings}\n\n"
                        "Evaluate match quality."
                    ),
                ),
            ]
        )

        # ---- chains --------------------------------------------------
        self.generation_chain = self.job_generation_prompt | self.llm
        self.matching_chain = self.matching_prompt | self.llm

    # ================================================================ #
    #  Public API
    # ================================================================ #

    def search_jobs(
        self,
        query: str,
        location: str,
        experience_level: str = "mid",
        remote_ok: bool = True,
        count: int = 5,
    ) -> Dict[str, Any]:
        """Run a full job search: geocode → find nearby offices → generate listings.

        Parameters
        ----------
        query : str
            Role or keywords (e.g. "Python backend engineer").
        location : str
            City or region name (e.g. "Tel Aviv", "Berlin").
        experience_level : str
            One of junior / mid / senior / lead.
        remote_ok : bool
            Whether to include remote positions.
        count : int
            Number of job listings to generate (1–10).

        Returns
        -------
        dict
            ``jobs`` – list of job dicts,
            ``total_found`` – count,
            ``search_query``, ``location``,
            ``location_info`` – geocoding result,
            ``nearby_companies`` – Overpass results,
            ``processing_time``.
        """
        start = time.time()
        count = max(1, min(count, 10))

        # ---- Step 1: Geocode the location ----------------------------
        location_info = self.get_city_center(location)

        # ---- Step 2: Find nearby tech offices via Overpass -----------
        nearby_companies: List[Dict[str, Any]] = []
        if location_info.get("lat") and location_info.get("lon"):
            nearby_companies = self.find_nearby_offices(
                lat=location_info["lat"],
                lon=location_info["lon"],
                radius=5000,
            )

        # ---- Step 3: Generate job listings via LLM -------------------
        jobs = self._generate_job_listings(
            query=query,
            location=location,
            experience_level=experience_level,
            remote_ok=remote_ok,
            nearby_companies=nearby_companies,
            location_info=location_info,
            count=count,
        )

        elapsed = round(time.time() - start, 3)
        return {
            "jobs": jobs,
            "total_found": len(jobs),
            "search_query": query,
            "location": location,
            "location_info": location_info,
            "nearby_companies": nearby_companies[:5],
            "processing_time": elapsed,
        }

    def search_jobs_with_matching(
        self,
        query: str,
        location: str,
        experience_level: str = "mid",
        remote_ok: bool = True,
        user_profile: Optional[Dict[str, Any]] = None,
        count: int = 5,
    ) -> Dict[str, Any]:
        """Search for jobs and score them against a user profile."""
        # First get raw search results
        search_results = self.search_jobs(
            query=query,
            location=location,
            experience_level=experience_level,
            remote_ok=remote_ok,
            count=count,
        )

        # If no profile provided, return raw results
        if not user_profile:
            return search_results

        # Score jobs against user profile
        matching = self._match_jobs_to_profile(
            jobs=search_results["jobs"],
            user_profile=user_profile,
        )

        search_results["matching_analysis"] = matching
        return search_results

    # ================================================================ #
    #  Geolocation helpers
    # ================================================================ #

    def get_city_center(self, city: str) -> Dict[str, Any]:
        """Geocode a city name to lat/lon using OpenStreetMap Nominatim.

        Returns
        -------
        dict
            ``city``, ``lat``, ``lon``, ``display_name``, ``country``, ``found``.
        """
        try:
            resp = requests.get(
                NOMINATIM_URL,
                params={
                    "q": city,
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 1,
                },
                headers={"User-Agent": _USER_AGENT},
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data:
                logger.warning("Nominatim returned no results for '%s'", city)
                return {"city": city, "lat": None, "lon": None, "found": False}

            result = data[0]
            address = result.get("address", {})
            return {
                "city": city,
                "lat": float(result["lat"]),
                "lon": float(result["lon"]),
                "display_name": result.get("display_name", ""),
                "country": address.get("country", ""),
                "country_code": address.get("country_code", ""),
                "found": True,
            }

        except requests.RequestException as exc:
            logger.error("Nominatim geocoding failed for '%s': %s", city, exc)
            return {"city": city, "lat": None, "lon": None, "found": False}

    def find_nearby_offices(
        self,
        lat: float,
        lon: float,
        radius: int = 5000,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query Overpass API for tech companies / offices near coordinates.

        Parameters
        ----------
        lat, lon : float
            Centre coordinates.
        radius : int
            Search radius in metres (default 5 km).
        limit : int
            Maximum results to return.

        Returns
        -------
        list[dict]
            Each dict: ``name``, ``type``, ``lat``, ``lon``, ``distance_m``.
        """
        overpass_query = f"""
        [out:json][timeout:10];
        (
          node["office"~"company|it|coworking"](around:{radius},{lat},{lon});
          node["amenity"="coworking_space"](around:{radius},{lat},{lon});
          way["office"~"company|it|coworking"](around:{radius},{lat},{lon});
        );
        out center {limit};
        """

        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": overpass_query},
                headers={"User-Agent": _USER_AGENT},
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            results: List[Dict[str, Any]] = []
            for element in data.get("elements", []):
                tags = element.get("tags", {})
                name = tags.get("name", tags.get("operator", "Unknown Office"))

                # Coordinates: nodes have lat/lon directly, ways use 'center'
                e_lat = element.get("lat") or element.get("center", {}).get("lat")
                e_lon = element.get("lon") or element.get("center", {}).get("lon")

                if not e_lat or not e_lon:
                    continue

                distance = self._haversine(lat, lon, e_lat, e_lon)
                office_type = tags.get("office", tags.get("amenity", "office"))

                results.append(
                    {
                        "name": name,
                        "type": office_type,
                        "lat": round(e_lat, 6),
                        "lon": round(e_lon, 6),
                        "distance_m": round(distance),
                        "address": tags.get("addr:street", ""),
                    }
                )

            # Sort by distance
            results.sort(key=lambda x: x["distance_m"])
            return results[:limit]

        except requests.RequestException as exc:
            logger.error("Overpass API query failed: %s", exc)
            return []

    # ================================================================ #
    #  Internal helpers
    # ================================================================ #

    def _generate_job_listings(
        self,
        query: str,
        location: str,
        experience_level: str,
        remote_ok: bool,
        nearby_companies: List[Dict[str, Any]],
        location_info: Dict[str, Any],
        count: int,
    ) -> List[Dict[str, Any]]:
        """Generate realistic job listings via LLM."""
        # Format nearby companies for context
        nearby_text = "None found"
        if nearby_companies:
            nearby_items = [
                f"- {c['name']} ({c['type']}, {c['distance_m']}m away)"
                for c in nearby_companies[:8]
            ]
            nearby_text = "\n".join(nearby_items)

        # Format location info
        loc_text = "Location not geocoded"
        if location_info.get("found"):
            loc_text = (
                f"{location_info.get('display_name', location)} "
                f"(lat: {location_info['lat']}, lon: {location_info['lon']}, "
                f"country: {location_info.get('country', 'unknown')})"
            )

        try:
            response = self.generation_chain.invoke(
                {
                    "query": query,
                    "location": location,
                    "experience_level": experience_level,
                    "remote_ok": str(remote_ok),
                    "nearby_companies": nearby_text,
                    "location_info": loc_text,
                    "count": str(count),
                }
            )
            jobs = self._parse_llm_json(response.content)

            if isinstance(jobs, list):
                return self._normalize_job_listings(jobs)
            elif isinstance(jobs, dict) and "jobs" in jobs:
                return self._normalize_job_listings(jobs["jobs"])
            else:
                logger.warning("LLM returned unexpected structure for jobs")
                return self._fallback_listings(query, location, experience_level, remote_ok)

        except Exception as exc:
            logger.error("Job generation LLM call failed: %s", exc)
            return self._fallback_listings(query, location, experience_level, remote_ok)

    def _match_jobs_to_profile(
        self,
        jobs: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Score job listings against a user profile via LLM."""
        try:
            response = self.matching_chain.invoke(
                {
                    "user_profile": json.dumps(user_profile, default=str),
                    "job_listings": json.dumps(jobs, default=str),
                }
            )
            return self._parse_llm_json(response.content)
        except Exception as exc:
            logger.error("Job matching failed: %s", exc)
            return None

    def _normalize_job_listings(self, raw_jobs: List[Any]) -> List[Dict[str, Any]]:
        """Ensure every job dict has the expected keys."""
        normalized: List[Dict[str, Any]] = []
        for item in raw_jobs:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "title": item.get("title", "Unknown Role"),
                    "company": item.get("company", "Unknown Company"),
                    "location": item.get("location", ""),
                    "description": item.get("description", ""),
                    "url": item.get("url"),
                    "salary_range": item.get("salary_range"),
                    "remote_friendly": bool(item.get("remote_friendly", False)),
                    "match_score": float(item.get("match_score", 0.5)),
                    "experience_level": item.get("experience_level", "mid"),
                    "key_skills": item.get("key_skills", []),
                }
            )
        return normalized

    def _fallback_listings(
        self,
        query: str,
        location: str,
        experience_level: str,
        remote_ok: bool,
    ) -> List[Dict[str, Any]]:
        """Produce safe fallback listings when the LLM fails."""
        level_label = experience_level.title()
        return [
            {
                "title": f"{level_label} {query}",
                "company": "Tech Company",
                "location": location,
                "description": (
                    f"Seeking a motivated {experience_level}-level professional "
                    f"for a {query} role. Competitive salary and benefits."
                ),
                "url": None,
                "salary_range": "Competitive",
                "remote_friendly": remote_ok,
                "match_score": 0.5,
                "experience_level": experience_level,
                "key_skills": [],
            },
            {
                "title": f"{query} — {level_label}",
                "company": "Growing Startup",
                "location": "Remote" if remote_ok else location,
                "description": (
                    f"Join our team as a {query}. "
                    f"We're looking for {experience_level}-level talent to help us scale."
                ),
                "url": None,
                "salary_range": "Competitive",
                "remote_friendly": True if remote_ok else False,
                "match_score": 0.4,
                "experience_level": experience_level,
                "key_skills": [],
            },
        ]

    # ---- geometry ---------------------------------------------------

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in metres between two points."""
        import math

        R = 6_371_000  # Earth radius in metres
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(d_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # ---- JSON parsing ------------------------------------------------

    @staticmethod
    def _parse_llm_json(text: str) -> Any:
        """Best-effort JSON extraction from LLM output."""
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = cleaned.strip().rstrip("`")

        # Direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try finding outermost array
        bracket_start = cleaned.find("[")
        bracket_end = cleaned.rfind("]")
        if bracket_start != -1 and bracket_end > bracket_start:
            try:
                return json.loads(cleaned[bracket_start : bracket_end + 1])
            except json.JSONDecodeError:
                pass

        # Try finding outermost object
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(cleaned[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse LLM JSON:\n%s", text[:300])
        return None
