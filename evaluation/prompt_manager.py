"""
Prompt Manager — versioned prompt loading, comparison, and hot-swap.

Loads prompt definitions from ``evaluation/prompts/<agent>_<version>.json``
and can inject them into agent instances for A/B testing or regression runs.

Usage::

    from evaluation.prompt_manager import PromptManager

    pm = PromptManager()
    pm.list_versions("resume")         # ["v1.0"]
    prompts = pm.load("resume", "v1.0")
    pm.compare("resume", "v1.0", "v1.1")
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class PromptManager:
    """Load and manage versioned prompt files."""

    def __init__(self, prompts_dir: str | Path | None = None):
        self.prompts_dir = Path(prompts_dir) if prompts_dir else _PROMPTS_DIR

    # ------------------------------------------------------------------ #
    #  Discovery
    # ------------------------------------------------------------------ #

    def list_agents(self) -> List[str]:
        """Return agent names that have at least one prompt file."""
        agents: set[str] = set()
        for f in self.prompts_dir.glob("*.json"):
            match = re.match(r"^(.+?)_v[\d.]+\.json$", f.name)
            if match:
                agents.add(match.group(1))
        return sorted(agents)

    def list_versions(self, agent: str) -> List[str]:
        """Return sorted version strings for *agent*."""
        versions: List[str] = []
        for f in self.prompts_dir.glob(f"{agent}_v*.json"):
            match = re.search(r"_v([\d.]+)\.json$", f.name)
            if match:
                versions.append(f"v{match.group(1)}")
        return sorted(versions)

    def latest_version(self, agent: str) -> str:
        """Return the latest (highest) version for *agent*."""
        vs = self.list_versions(agent)
        if not vs:
            raise FileNotFoundError(f"No prompt files for agent '{agent}'")
        return vs[-1]

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #

    def load(self, agent: str, version: str | None = None) -> Dict[str, Any]:
        """Load prompt file for *agent* and *version*.

        Parameters
        ----------
        agent:
            Agent name (``resume``, ``interview``, etc.)
        version:
            Prompt version string like ``v1.0``.  If ``None``, loads the
            latest available version.

        Returns
        -------
        dict
            Full JSON contents of the prompt file.
        """
        version = version or self.latest_version(agent)
        path = self.prompts_dir / f"{agent}_{version}.json"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.debug("Loaded prompt %s/%s from %s", agent, version, path)
        return data

    def load_prompt_text(
        self,
        agent: str,
        prompt_name: str,
        role: str = "system",
        version: str | None = None,
    ) -> str:
        """Load a single prompt text string.

        Parameters
        ----------
        agent:
            Agent name.
        prompt_name:
            Prompt function name inside the version file (e.g. ``analysis``).
        role:
            ``system`` or ``human``.
        version:
            Prompt version. ``None`` = latest.

        Returns
        -------
        str
            The raw prompt template text.
        """
        data = self.load(agent, version)
        prompts_section = data.get("prompts", {})
        prompt_def = prompts_section.get(prompt_name)
        if prompt_def is None:
            raise KeyError(
                f"Prompt '{prompt_name}' not found in {agent} {version}. "
                f"Available: {list(prompts_section.keys())}"
            )
        return prompt_def.get(role, "")

    # ------------------------------------------------------------------ #
    #  Comparison
    # ------------------------------------------------------------------ #

    def compare(
        self,
        agent: str,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """Compare two prompt versions and return a diff summary.

        Returns
        -------
        dict
            Keys: ``agent``, ``version_a``, ``version_b``, ``diffs`` (per-prompt changes).
        """
        data_a = self.load(agent, version_a)
        data_b = self.load(agent, version_b)

        prompts_a = data_a.get("prompts", {})
        prompts_b = data_b.get("prompts", {})

        all_keys = sorted(set(prompts_a.keys()) | set(prompts_b.keys()))

        diffs: Dict[str, Any] = {}
        for key in all_keys:
            pa = prompts_a.get(key, {})
            pb = prompts_b.get(key, {})
            if pa == pb:
                diffs[key] = {"changed": False}
            else:
                diffs[key] = {
                    "changed": True,
                    "system_changed": pa.get("system") != pb.get("system"),
                    "human_changed": pa.get("human") != pb.get("human"),
                    "a_system_len": len(pa.get("system", "")),
                    "b_system_len": len(pb.get("system", "")),
                }

        return {
            "agent": agent,
            "version_a": version_a,
            "version_b": version_b,
            "diffs": diffs,
            "a_metadata": data_a.get("metadata", {}),
            "b_metadata": data_b.get("metadata", {}),
        }

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #

    def validate(self, agent: str, version: str | None = None) -> List[str]:
        """Validate prompt file structure.  Returns list of issues (empty = OK)."""
        issues: List[str] = []
        try:
            data = self.load(agent, version)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            return [str(exc)]

        if "version" not in data:
            issues.append("Missing 'version' key")
        if "agent" not in data:
            issues.append("Missing 'agent' key")
        elif data["agent"] != agent:
            issues.append(f"Agent mismatch: file says '{data['agent']}', expected '{agent}'")
        if "prompts" not in data or not isinstance(data.get("prompts"), dict):
            issues.append("Missing or invalid 'prompts' dict")
        else:
            for pname, pdef in data["prompts"].items():
                if not isinstance(pdef, dict):
                    issues.append(f"Prompt '{pname}' is not a dict")
                elif "system" not in pdef:
                    issues.append(f"Prompt '{pname}' missing 'system' key")

        return issues

    def validate_all(self) -> Dict[str, List[str]]:
        """Validate every discovered prompt file.  Returns agent→issues mapping."""
        results: Dict[str, List[str]] = {}
        for agent in self.list_agents():
            for version in self.list_versions(agent):
                key = f"{agent}/{version}"
                results[key] = self.validate(agent, version)
        return results
