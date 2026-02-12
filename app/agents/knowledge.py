"""
Knowledge Agent — RAG-powered career-advice Q&A.

Uses the ChromaDB career-guides collection to retrieve relevant context,
then feeds that context + the user question to GPT-4o-mini for a grounded,
source-attributed answer.

Usage::

    from app.agents.knowledge import KnowledgeAgent

    agent = KnowledgeAgent()
    result = agent.answer_question("How should I prepare for a system design interview?")
    print(result["answer"])
    print(result["sources"])
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.rag.query import query_knowledge_base, get_formatted_context

logger = logging.getLogger(__name__)


class KnowledgeAgent:
    """Retrieval-Augmented Generation agent for career coaching questions."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_context_chunks: int = 5,
    ):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.max_context_chunks = max_context_chunks

        # ---- QA prompt ------------------------------------------------
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an expert career coach and job-search advisor. "
                        "Answer the user's question using ONLY the provided context. "
                        "If the context does not contain enough information, say so honestly "
                        "and offer your best general advice, clearly marking it as general guidance.\n\n"
                        "RULES:\n"
                        "1. Ground every claim in the context where possible.\n"
                        "2. Reference sources by their names when citing information.\n"
                        "3. Be actionable — give concrete steps, not vague platitudes.\n"
                        "4. Keep the answer focused and well-structured.\n"
                        "5. If the question is not career-related, politely redirect.\n\n"
                        "Return your response as a JSON object with EXACTLY this schema:\n"
                        "{{\n"
                        '  "answer": "<your detailed answer>",\n'
                        '  "sources_used": ["<source name 1>", "<source name 2>"],\n'
                        '  "confidence": <float 0.0-1.0>,\n'
                        '  "related_topics": ["<topic 1>", "<topic 2>", "<topic 3>"]\n'
                        "}}\n\n"
                        "Return ONLY valid JSON — no markdown fences, no extra text."
                    ),
                ),
                (
                    "human",
                    (
                        "CONTEXT (retrieved from knowledge base):\n"
                        "---\n"
                        "{context}\n"
                        "---\n\n"
                        "QUESTION: {question}"
                    ),
                ),
            ]
        )

        # ---- chain -----------------------------------------------------
        self.chain = self.qa_prompt | self.llm

    # ------------------------------------------------------------------ #
    #  Public methods
    # ------------------------------------------------------------------ #

    def answer_question(
        self,
        question: str,
        k: int | None = None,
    ) -> Dict[str, Any]:
        """Answer a career-coaching question using RAG.

        Parameters
        ----------
        question:
            The user's natural-language question.
        k:
            Number of context chunks to retrieve (defaults to
            ``self.max_context_chunks``).

        Returns
        -------
        dict
            ``answer``           – the generated answer text
            ``sources``          – list of source document names
            ``relevance_score``  – 0-1 confidence / relevance score
            ``related_topics``   – suggested follow-up topics
            ``context_chunks``   – number of chunks used
            ``processing_time``  – seconds elapsed
        """
        start = time.time()
        k = k or self.max_context_chunks

        # ---- retrieve --------------------------------------------------
        rag_results = query_knowledge_base(question, k=k)
        context_text = self._build_context(rag_results)
        has_context = bool(rag_results)

        # ---- generate --------------------------------------------------
        try:
            response = self.chain.invoke(
                {"context": context_text or "No relevant context found.", "question": question}
            )
            parsed = self._parse_llm_json(response.content)
        except Exception as exc:
            logger.error("LLM call failed in KnowledgeAgent: %s", exc)
            parsed = None

        # ---- assemble result -------------------------------------------
        if parsed:
            answer = parsed.get("answer", "I could not generate an answer.")
            sources_used = parsed.get("sources_used", [])
            confidence = float(parsed.get("confidence", 0.5 if has_context else 0.3))
            related_topics = parsed.get("related_topics", [])
        else:
            answer, sources_used, confidence, related_topics = self._fallback_answer(
                question, rag_results
            )

        # De-duplicate sources: merge RAG source names + LLM-cited names
        all_sources = self._merge_sources(rag_results, sources_used)

        elapsed = round(time.time() - start, 3)
        return {
            "answer": answer,
            "sources": all_sources,
            "relevance_score": round(confidence, 2),
            "related_topics": related_topics[:5],
            "context_chunks": len(rag_results),
            "processing_time": elapsed,
        }

    def get_topic_summary(self, topic: str) -> Dict[str, Any]:
        """Retrieve a short knowledge-base summary for a single topic.

        This is lighter-weight than :meth:`answer_question` — it returns
        the raw retrieved chunks rather than running a full LLM call.
        Useful for enriching other agents' prompts with quick context.
        """
        start = time.time()
        results = query_knowledge_base(topic, k=3)
        chunks = [r["content"] for r in results]
        sources = list({r["source"] for r in results})
        avg_score = (
            round(sum(r["score"] for r in results) / len(results), 4)
            if results
            else 0.0
        )
        return {
            "topic": topic,
            "chunks": chunks,
            "sources": sources,
            "avg_relevance": avg_score,
            "processing_time": round(time.time() - start, 3),
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_context(rag_results: List[Dict[str, Any]]) -> str:
        """Format RAG results into a numbered context block."""
        if not rag_results:
            return ""
        sections: List[str] = []
        for i, r in enumerate(rag_results, 1):
            sections.append(f"[Source {i}: {r['source']}]\n{r['content']}")
        return "\n\n---\n\n".join(sections)

    @staticmethod
    def _parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
        """Best-effort extraction of JSON from LLM output."""
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = cleaned.strip().rstrip("`")

        # Try direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try extracting the outermost { … }
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(cleaned[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse LLM JSON:\n%s", text[:300])
        return None

    @staticmethod
    def _fallback_answer(
        question: str, rag_results: List[Dict[str, Any]]
    ) -> tuple:
        """Produce a safe fallback when the LLM call or parsing fails."""
        if rag_results:
            # Stitch together the top chunks as a best-effort answer
            combined = "\n\n".join(r["content"] for r in rag_results[:3])
            answer = (
                "I found some relevant information in our knowledge base, "
                "though I wasn't able to synthesise a full answer:\n\n"
                f"{combined}"
            )
            sources = list({r["source"] for r in rag_results})
            confidence = 0.4
        else:
            answer = (
                "I don't have specific information about that topic in my "
                "knowledge base right now. Please try rephrasing your "
                "question or ask about resume writing, interviews, salary "
                "negotiation, or industry trends."
            )
            sources = []
            confidence = 0.1

        related_topics = [
            "resume best practices",
            "interview preparation",
            "salary negotiation",
        ]
        return answer, sources, confidence, related_topics

    @staticmethod
    def _merge_sources(
        rag_results: List[Dict[str, Any]], llm_sources: List[str]
    ) -> List[str]:
        """Combine RAG-retrieved source names with LLM-cited source names,
        preserving order and removing duplicates."""
        seen: set = set()
        merged: List[str] = []

        # RAG sources first (highest authority)
        for r in rag_results:
            name = r["source"]
            if name.lower() not in seen:
                seen.add(name.lower())
                merged.append(name)

        # Then any additional names the LLM cited
        for name in llm_sources:
            if name.lower() not in seen:
                seen.add(name.lower())
                merged.append(name)

        return merged
