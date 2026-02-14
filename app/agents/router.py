"""
Router Agent — LLM-powered intent classification for query routing.

Classifies user queries into one of the known intent types and returns a
confidence score.  Falls back to keyword-based classification when the LLM
is unavailable or returns an unparseable result.

Usage::

    from app.agents.router import RouterAgent

    router = RouterAgent()
    result = router.classify_intent("Help me improve my resume")
    # {
    #     "intent": "resume_improvement",
    #     "confidence": 0.95,
    #     "reasoning": "User explicitly asks to improve their resume.",
    #     "classification_method": "llm"
    # }
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import INTENT_TYPES

logger = logging.getLogger(__name__)

# Minimum confidence to trust the LLM classification; below this we
# fall back to the keyword classifier.
CONFIDENCE_THRESHOLD = 0.7

# Map intents → the downstream graph node name
INTENT_TO_AGENT = {
    "resume_analysis": "resume",
    "resume_improvement": "resume",
    "interview_practice": "interview",
    "interview_start": "interview",
    "interview_answer": "interview",
    "job_search": "job_search",
    "career_advice": "knowledge",
    "application_tracking": "knowledge",
    "general_question": "knowledge",
    "unknown": "knowledge",
}


class RouterAgent:
    """LLM-powered intent classifier for the multi-agent workflow.

    Parameters
    ----------
    model : str
        OpenAI model name.  Default ``gpt-4o-mini``.
    temperature : float
        Sampling temperature.  ``0.0`` for deterministic classification.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self.llm = ChatOpenAI(model=model, temperature=temperature)

        # ---- classification prompt -----------------------------------
        self.classification_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are an intent classifier for an AI career coaching assistant.\n\n"
                    "Given a user query (and optional context), classify the user's intent "
                    "into EXACTLY ONE of the following categories:\n\n"
                    "{intent_list}\n\n"
                    "Category definitions:\n"
                    "- resume_analysis: User wants their resume reviewed, scored, or analyzed\n"
                    "- resume_improvement: User wants specific suggestions to improve their resume\n"
                    "- interview_practice: User wants to practice interview questions or get feedback\n"
                    "- interview_start: User explicitly wants to begin a new mock interview session\n"
                    "- interview_answer: User is answering an interview question in an active session\n"
                    "- job_search: User is looking for job opportunities, openings, or companies\n"
                    "- career_advice: User seeks general career guidance, tips, or strategies\n"
                    "- application_tracking: User wants to track, update, or check job applications\n"
                    "- general_question: General question about the system or non-specific query\n"
                    "- unknown: Cannot determine intent from the query\n\n"
                    "Return ONLY valid JSON with this exact structure:\n"
                    '{{\n'
                    '  "intent": "<one of the categories above>",\n'
                    '  "confidence": <float between 0.0 and 1.0>,\n'
                    '  "reasoning": "<one sentence explaining your classification>"\n'
                    '}}\n\n'
                    "Rules:\n"
                    "- If the query mentions an active interview session ID, classify as interview_answer\n"
                    "- If the query is ambiguous, prefer career_advice over unknown\n"
                    "- Confidence should reflect how certain you are (0.9+ for very clear, 0.5-0.7 for ambiguous)\n"
                    "- Return ONLY the JSON object — no markdown fences, no extra text"
                ),
            ),
            (
                "human",
                (
                    "User query: {user_query}\n\n"
                    "User context (if available): {user_context}\n\n"
                    "Active interview session: {has_active_session}"
                ),
            ),
        ])

        self.classification_chain = self.classification_prompt | self.llm

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def classify_intent(
        self,
        user_query: str,
        user_context: Optional[Dict[str, Any]] = None,
        has_active_session: bool = False,
    ) -> Dict[str, Any]:
        """Classify the intent of *user_query*.

        Parameters
        ----------
        user_query : str
            Raw text from the user.
        user_context : dict, optional
            User profile / previous conversation summary for context-aware
            routing.
        has_active_session : bool
            If ``True``, an interview session is in progress — biases
            toward ``interview_answer``.

        Returns
        -------
        dict
            Keys: ``intent``, ``confidence``, ``reasoning``,
            ``classification_method``, ``processing_time``.
        """
        start_time = time.time()

        # --- fast-path: active interview session ----------------------
        if has_active_session and not self._is_explicit_new_intent(user_query):
            return {
                "intent": "interview_answer",
                "confidence": 0.95,
                "reasoning": "Active interview session detected; treating input as an answer.",
                "classification_method": "session_override",
                "processing_time": time.time() - start_time,
            }

        # --- LLM classification ---------------------------------------
        try:
            result = self._llm_classify(user_query, user_context, has_active_session)
            result["processing_time"] = time.time() - start_time

            # If LLM confidence is below threshold, cross-check with keyword
            if result["confidence"] < CONFIDENCE_THRESHOLD:
                kw_result = self._keyword_classify(user_query)
                if kw_result["confidence"] > result["confidence"]:
                    logger.info(
                        "LLM confidence %.2f below threshold; using keyword result '%s' (%.2f)",
                        result["confidence"],
                        kw_result["intent"],
                        kw_result["confidence"],
                    )
                    kw_result["classification_method"] = "keyword_fallback"
                    kw_result["processing_time"] = time.time() - start_time
                    return kw_result

            return result

        except Exception as exc:
            logger.warning("LLM classification failed (%s); falling back to keywords", exc)
            kw_result = self._keyword_classify(user_query)
            kw_result["classification_method"] = "keyword_fallback"
            kw_result["processing_time"] = time.time() - start_time
            return kw_result

    @staticmethod
    def resolve_agent(intent: str) -> str:
        """Map an intent string to the downstream graph node name."""
        return INTENT_TO_AGENT.get(intent, "knowledge")

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _llm_classify(
        self,
        user_query: str,
        user_context: Optional[Dict[str, Any]],
        has_active_session: bool,
    ) -> Dict[str, Any]:
        """Call the LLM and parse the structured response."""
        context_str = json.dumps(user_context, default=str) if user_context else "None"

        raw = self.classification_chain.invoke({
            "user_query": user_query,
            "user_context": context_str,
            "has_active_session": str(has_active_session),
            "intent_list": "\n".join(f"- {it}" for it in INTENT_TYPES),
        })

        parsed = self._parse_llm_json(raw.content)

        # Validate intent is in our known list
        intent = parsed.get("intent", "unknown")
        if intent not in INTENT_TYPES:
            logger.warning("LLM returned unknown intent '%s'; mapping to 'unknown'", intent)
            intent = "unknown"

        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # clamp

        return {
            "intent": intent,
            "confidence": confidence,
            "reasoning": parsed.get("reasoning", ""),
            "classification_method": "llm",
        }

    @staticmethod
    def _keyword_classify(user_query: str) -> Dict[str, Any]:
        """Deterministic keyword-based fallback classifier."""
        query_lower = user_query.lower()
        intent = "unknown"
        confidence = 0.5

        keyword_map = [
            # More specific patterns MUST come before broader ones
            (["improve my resume", "rewrite", "bullet points", "better resume"], "resume_improvement", 0.85),
            (["resume", "cv", "review my", "analyze my"], "resume_analysis", 0.85),
            (["start interview", "begin interview", "new interview session"], "interview_start", 0.90),
            (["interview", "practice", "mock interview", "prepare for interview"], "interview_practice", 0.85),
            (["job", "search", "find job", "opportunities", "openings", "hiring"], "job_search", 0.85),
            (["advice", "tips", "guide", "how to", "how do i", "negotiate", "salary"], "career_advice", 0.80),
            (["application", "track", "status", "applied", "follow up"], "application_tracking", 0.85),
        ]

        for keywords, mapped_intent, mapped_confidence in keyword_map:
            if any(kw in query_lower for kw in keywords):
                intent = mapped_intent
                confidence = mapped_confidence
                break

        return {
            "intent": intent,
            "confidence": confidence,
            "reasoning": f"Keyword match on query '{user_query[:80]}'",
            "classification_method": "keyword",
        }

    @staticmethod
    def _is_explicit_new_intent(user_query: str) -> bool:
        """Check if the query explicitly starts a new intent despite an active session."""
        new_intent_signals = [
            "search for job",
            "find job",
            "review my resume",
            "analyze my resume",
            "career advice",
            "help me with",
            "i want to",
            "switch to",
            "stop interview",
            "end session",
            "start new",
        ]
        query_lower = user_query.lower()
        return any(signal in query_lower for signal in new_intent_signals)

    @staticmethod
    def _parse_llm_json(content: str) -> Dict[str, Any]:
        """Parse JSON from LLM output, stripping markdown fences if present."""
        text = content.strip()

        # Strip ```json ... ``` fences
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract the first JSON object
            obj_match = re.search(r"\{.*\}", text, re.DOTALL)
            if obj_match:
                try:
                    return json.loads(obj_match.group())
                except json.JSONDecodeError:
                    pass

        logger.warning("Failed to parse LLM JSON: %s", text[:200])
        return {"intent": "unknown", "confidence": 0.0, "reasoning": "Parse error"}
