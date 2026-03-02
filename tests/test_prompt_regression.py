"""
Prompt regression tests — ensures prompt files are well-formed and
that key invariants hold across versions.

Run with::

    pytest tests/test_prompt_regression.py -v
"""

import json
import pytest
from pathlib import Path

from evaluation.prompt_manager import PromptManager


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def pm() -> PromptManager:
    return PromptManager()


EXPECTED_AGENTS = ["resume", "interview", "knowledge", "job_search", "router"]


# ------------------------------------------------------------------ #
#  Discovery tests
# ------------------------------------------------------------------ #

class TestPromptDiscovery:
    """All expected agents have at least one prompt version file."""

    def test_all_agents_present(self, pm: PromptManager):
        agents = pm.list_agents()
        for expected in EXPECTED_AGENTS:
            assert expected in agents, f"Missing prompt file for agent '{expected}'"

    def test_all_agents_have_v1(self, pm: PromptManager):
        for agent in EXPECTED_AGENTS:
            versions = pm.list_versions(agent)
            assert "v1.0" in versions, f"Agent '{agent}' missing v1.0 baseline"

    def test_latest_version_returns_string(self, pm: PromptManager):
        for agent in EXPECTED_AGENTS:
            v = pm.latest_version(agent)
            assert isinstance(v, str) and v.startswith("v")


# ------------------------------------------------------------------ #
#  Structure tests
# ------------------------------------------------------------------ #

class TestPromptStructure:
    """Every prompt file has valid JSON structure."""

    def test_all_files_valid(self, pm: PromptManager):
        issues = pm.validate_all()
        for key, problems in issues.items():
            assert problems == [], f"{key} has issues: {problems}"

    @pytest.mark.parametrize("agent", EXPECTED_AGENTS)
    def test_prompt_has_system_key(self, pm: PromptManager, agent: str):
        data = pm.load(agent, "v1.0")
        for pname, pdef in data["prompts"].items():
            assert "system" in pdef, f"{agent}/v1.0/{pname} missing 'system'"
            assert len(pdef["system"]) > 50, f"{agent}/v1.0/{pname} system prompt too short"


# ------------------------------------------------------------------ #
#  Content invariant tests
# ------------------------------------------------------------------ #

class TestPromptInvariants:
    """Key content invariants that should hold across ALL prompt versions."""

    def test_resume_analysis_requires_json(self, pm: PromptManager):
        text = pm.load_prompt_text("resume", "analysis", "system")
        assert "JSON" in text, "Resume analysis prompt must mention JSON output"
        assert "overall_score" in text, "Resume analysis must request overall_score"

    def test_resume_analysis_has_ats(self, pm: PromptManager):
        text = pm.load_prompt_text("resume", "analysis", "system")
        assert "ats" in text.lower(), "Resume analysis must cover ATS"

    def test_interview_eval_has_star(self, pm: PromptManager):
        text = pm.load_prompt_text("interview", "evaluation", "system")
        assert "STAR" in text, "Interview evaluation should reference STAR method"

    def test_router_has_all_intents(self, pm: PromptManager):
        text = pm.load_prompt_text("router", "classification", "system")
        required_intents = [
            "resume_analysis", "resume_improvement", "interview_practice",
            "job_search", "career_advice", "general_question",
        ]
        for intent in required_intents:
            assert intent in text, f"Router prompt missing intent '{intent}'"

    def test_knowledge_requires_grounding(self, pm: PromptManager):
        text = pm.load_prompt_text("knowledge", "qa", "system")
        assert "context" in text.lower(), "Knowledge prompt must reference context"
        assert "source" in text.lower(), "Knowledge prompt must reference sources"

    def test_job_search_has_salary(self, pm: PromptManager):
        text = pm.load_prompt_text("job_search", "job_generation", "system")
        assert "salary" in text.lower(), "Job search prompt must mention salary"


# ------------------------------------------------------------------ #
#  Cross-version regression tests
# ------------------------------------------------------------------ #

class TestCrossVersionRegression:
    """When multiple versions exist, verify that newer versions don't
    drop critical content."""

    @pytest.mark.parametrize("agent", EXPECTED_AGENTS)
    def test_newer_version_has_same_prompt_keys(self, pm: PromptManager, agent: str):
        versions = pm.list_versions(agent)
        if len(versions) < 2:
            pytest.skip(f"Only one version for {agent}")

        baseline = pm.load(agent, versions[0])
        latest = pm.load(agent, versions[-1])

        baseline_keys = set(baseline.get("prompts", {}).keys())
        latest_keys = set(latest.get("prompts", {}).keys())

        missing = baseline_keys - latest_keys
        assert not missing, (
            f"Agent '{agent}' {versions[-1]} dropped prompts: {missing}"
        )

    @pytest.mark.parametrize("agent", EXPECTED_AGENTS)
    def test_system_prompts_not_empty(self, pm: PromptManager, agent: str):
        for version in pm.list_versions(agent):
            data = pm.load(agent, version)
            for pname, pdef in data["prompts"].items():
                sys = pdef.get("system", "")
                assert len(sys) > 0, f"{agent}/{version}/{pname} has empty system prompt"
