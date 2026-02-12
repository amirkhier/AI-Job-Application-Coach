"""
Interview Agent — LLM-powered mock interview practice and evaluation.

This agent uses GPT-4o-mini to provide:
- Role-specific interview question generation (behavioral, technical, situational)
- Structured answer evaluation using the STAR method
- Per-question scoring with actionable feedback
- Full session summary with strengths / weaknesses across all answers
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List, Optional, Any
import json
import logging
import time

logger = logging.getLogger(__name__)


class InterviewAgent:
    """Generates interview questions and evaluates candidate answers.

    Usage::

        agent = InterviewAgent()
        questions = agent.generate_questions("Backend Engineer", "senior", count=5)
        feedback  = agent.evaluate_answer(questions[0], user_answer, "Backend Engineer", "senior")
        summary   = agent.generate_session_summary(questions, answers_with_feedback, "Backend Engineer", "senior")
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self._question_prompt = self._build_question_prompt()
        self._evaluation_prompt = self._build_evaluation_prompt()
        self._summary_prompt = self._build_summary_prompt()
        logger.info("InterviewAgent initialised (model=%s)", model)

    # ------------------------------------------------------------------ #
    #  Prompt builders
    # ------------------------------------------------------------------ #

    def _build_question_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are a seasoned hiring manager and interview coach.\n\n"
             "Generate exactly {count} interview questions for a **{level}-level "
             "{role}** position.\n\n"
             "Requirements:\n"
             "- Include a mix of question types:\n"
             "  • ~30-40 % behavioral\n"
             "  • ~40-50 % technical / role-specific\n"
             "  • ~20-30 % situational\n"
             "- Vary difficulty: include easy, medium, and hard questions.\n"
             "- Each question must be realistic and commonly used in real interviews.\n"
             "- Order questions from easier to harder.\n\n"
             "Return a **JSON array** where each element has exactly these keys:\n"
             "{{\n"
             '  "id": "q<number>",\n'
             '  "question": "<the question text>",\n'
             '  "type": "behavioral" | "technical" | "situational",\n'
             '  "difficulty": "easy" | "medium" | "hard",\n'
             '  "key_points": ["<point a good answer should cover>", "..."]\n'
             "}}\n\n"
             "Return ONLY the JSON array — no markdown fences, no extra text."),
            ("human",
             "Role: {role}\nLevel: {level}\nNumber of questions: {count}"),
        ])

    def _build_evaluation_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert interview coach evaluating a candidate's answer.\n\n"
             "Context:\n"
             "- Role: {role} ({level} level)\n"
             "- Question type: {question_type}\n\n"
             "Evaluate the answer on these dimensions (each scored 1-10):\n"
             "1. **Relevance** — Does it actually answer the question?\n"
             "2. **Depth & Specificity** — Concrete examples, metrics, details?\n"
             "3. **Structure** — Is the answer well-organised (e.g. STAR method)?\n"
             "4. **Communication** — Clarity, conciseness, professionalism?\n\n"
             "Return a **single JSON object** with exactly these keys:\n"
             "{{\n"
             '  "overall_score": <float 1.0-10.0>,\n'
             '  "dimension_scores": {{\n'
             '    "relevance": <float 1.0-10.0>,\n'
             '    "depth": <float 1.0-10.0>,\n'
             '    "structure": <float 1.0-10.0>,\n'
             '    "communication": <float 1.0-10.0>\n'
             "  }},\n"
             '  "strength_areas": ["<strength 1>", "..."],\n'
             '  "improvement_areas": ["<area to improve>", "..."],\n'
             '  "specific_feedback": "<2-3 sentence detailed, constructive feedback>",\n'
             '  "suggested_improvement": "<concrete advice on how to improve this specific answer>"\n'
             "}}\n\n"
             "Guidelines:\n"
             "- Be encouraging but honest.\n"
             "- Reference specific parts of the candidate's answer.\n"
             "- For behavioral questions, evaluate STAR usage.\n"
             "- For technical questions, evaluate accuracy and depth.\n"
             "- Suggested improvement should be actionable (e.g. a reworded opening sentence).\n\n"
             "Return ONLY the JSON object — no markdown fences, no extra text."),
            ("human",
             "Question:\n{question}\n\n"
             "Key points a good answer should cover:\n{key_points}\n\n"
             "Candidate's answer:\n{answer}"),
        ])

    def _build_summary_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are an interview coach providing a final session debrief.\n\n"
             "The candidate just completed a mock interview for a "
             "**{level}-level {role}** position.\n\n"
             "Below is a JSON array of each question, the candidate's answer, "
             "and the per-question evaluation.\n\n"
             "Produce a **single JSON object** with these keys:\n"
             "{{\n"
             '  "overall_score": <float 1.0-10.0 — weighted average across questions>,\n'
             '  "total_questions": <int>,\n'
             '  "performance_level": "needs_improvement" | "developing" | "competent" | "strong" | "exceptional",\n'
             '  "strongest_areas": ["<area>", "..."],\n'
             '  "weakest_areas": ["<area>", "..."],\n'
             '  "key_recommendations": [\n'
             '    "<actionable recommendation 1>",\n'
             '    "<up to 5 total>"\n'
             "  ],\n"
             '  "interview_readiness": "<1-2 sentence assessment of overall readiness>",\n'
             '  "next_steps": ["<suggested practice topic>", "..."]\n'
             "}}\n\n"
             "Be constructive and specific. Reference actual answers where relevant.\n\n"
             "Return ONLY the JSON object — no markdown fences, no extra text."),
            ("human", "{session_data}"),
        ])

    # ------------------------------------------------------------------ #
    #  JSON parsing helper (same pattern as ResumeAgent)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_llm_json(content: str) -> Any:
        """Parse JSON from LLM output, tolerating markdown fences."""
        text = content.strip()

        if text.startswith("```"):
            first_nl = text.find("\n")
            text = text[first_nl + 1:] if first_nl != -1 else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Attempt 1: direct
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Attempt 2: find outermost JSON structure
        # Could be an array or an object
        for open_ch, close_ch in [("{", "}"), ("[", "]")]:
            start = text.find(open_ch)
            end = text.rfind(close_ch) + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    continue

        logger.error("Failed to parse LLM JSON: %.300s…", text)
        raise ValueError("LLM returned invalid JSON — please retry")

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def generate_questions(
        self,
        role: str,
        level: str = "mid",
        count: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate role-specific interview questions.

        Args:
            role: Target job title (e.g. "Backend Engineer").
            level: Seniority — junior | mid | senior | lead.
            count: Number of questions to generate (1-10).

        Returns:
            List of question dicts, each with id, question, type,
            difficulty, and key_points.
        """
        start = time.time()

        try:
            chain = self._question_prompt | self.llm
            result = chain.invoke({
                "role": role,
                "level": level,
                "count": count,
            })

            questions = self._parse_llm_json(result.content)

            if not isinstance(questions, list):
                raise ValueError("Expected a JSON array of questions")

            # Normalise — ensure every question has an id
            for i, q in enumerate(questions):
                if "id" not in q:
                    q["id"] = f"q{i + 1}"

            elapsed = round(time.time() - start, 2)
            logger.info(
                "Generated %d questions for %s %s in %.2fs",
                len(questions), level, role, elapsed,
            )
            return questions

        except Exception as exc:
            logger.error("Question generation failed: %s", exc, exc_info=True)
            # Return a sensible fallback so the session can still start
            return [
                {
                    "id": f"q{i + 1}",
                    "question": q,
                    "type": "behavioral",
                    "difficulty": "easy",
                    "key_points": ["Clear communication", "Relevant experience"],
                }
                for i, q in enumerate([
                    f"Tell me about yourself and why you're interested in this {role} role.",
                    "Describe a challenging project you worked on. What was the outcome?",
                    "How do you handle tight deadlines and competing priorities?",
                    "What is your greatest professional achievement so far?",
                    "Where do you see yourself in the next 2-3 years?",
                ][:count])
            ]

    def evaluate_answer(
        self,
        question: Dict[str, Any],
        answer: str,
        role: str = "Software Engineer",
        level: str = "mid",
    ) -> Dict[str, Any]:
        """Evaluate a candidate's answer to an interview question.

        Args:
            question: The question dict (must have 'question', 'type', 'key_points').
            answer: The candidate's answer text.
            role: Target role for context.
            level: Seniority level for context.

        Returns:
            Dict with overall_score, dimension_scores, strength_areas,
            improvement_areas, specific_feedback, suggested_improvement,
            processing_time.
        """
        start = time.time()

        try:
            chain = self._evaluation_prompt | self.llm
            result = chain.invoke({
                "role": role,
                "level": level,
                "question_type": question.get("type", "behavioral"),
                "question": question.get("question", ""),
                "key_points": ", ".join(question.get("key_points", [])),
                "answer": answer,
            })

            evaluation = self._parse_llm_json(result.content)
            evaluation["processing_time"] = round(time.time() - start, 2)

            logger.info(
                "Answer evaluated in %.2fs — score: %s",
                evaluation["processing_time"],
                evaluation.get("overall_score", "N/A"),
            )
            return evaluation

        except ValueError:
            # JSON parse failure — return structured fallback
            elapsed = round(time.time() - start, 2)
            return {
                "overall_score": 5.0,
                "dimension_scores": {
                    "relevance": 5.0,
                    "depth": 5.0,
                    "structure": 5.0,
                    "communication": 5.0,
                },
                "strength_areas": ["Answer was provided"],
                "improvement_areas": ["Evaluation could not be fully completed"],
                "specific_feedback": "The system had difficulty evaluating this answer. Please try again.",
                "suggested_improvement": "Try structuring your answer using the STAR method: Situation, Task, Action, Result.",
                "processing_time": elapsed,
                "error": "LLM returned unparseable response",
            }
        except Exception as exc:
            logger.error("Answer evaluation failed: %s", exc, exc_info=True)
            raise

    def generate_session_summary(
        self,
        questions: List[Dict[str, Any]],
        answers_with_feedback: List[Dict[str, Any]],
        role: str = "Software Engineer",
        level: str = "mid",
    ) -> Dict[str, Any]:
        """Generate a comprehensive summary for a completed interview session.

        Args:
            questions: The full list of question dicts.
            answers_with_feedback: List of dicts, each with 'question_id',
                'answer', and 'evaluation'.
            role: Target role.
            level: Seniority level.

        Returns:
            Dict with overall_score, performance_level, strongest_areas,
            weakest_areas, key_recommendations, interview_readiness,
            next_steps, processing_time.
        """
        start = time.time()

        # Build a compact representation for the prompt
        session_data = []
        for q in questions:
            matching = next(
                (a for a in answers_with_feedback if a.get("question_id") == q["id"]),
                None,
            )
            entry = {
                "question": q.get("question"),
                "type": q.get("type"),
                "difficulty": q.get("difficulty"),
                "answer": matching.get("answer", "(no answer)") if matching else "(no answer)",
                "score": matching["evaluation"].get("overall_score") if matching and matching.get("evaluation") else None,
            }
            session_data.append(entry)

        try:
            chain = self._summary_prompt | self.llm
            result = chain.invoke({
                "role": role,
                "level": level,
                "session_data": json.dumps(session_data, indent=2),
            })

            summary = self._parse_llm_json(result.content)
            summary["processing_time"] = round(time.time() - start, 2)

            logger.info(
                "Session summary generated in %.2fs — overall: %s (%s)",
                summary["processing_time"],
                summary.get("overall_score", "N/A"),
                summary.get("performance_level", "N/A"),
            )
            return summary

        except ValueError:
            # Fallback: compute a basic summary from raw scores
            scores = [
                a["evaluation"]["overall_score"]
                for a in answers_with_feedback
                if a.get("evaluation") and a["evaluation"].get("overall_score")
            ]
            avg = round(sum(scores) / len(scores), 1) if scores else 0.0
            return {
                "overall_score": avg,
                "total_questions": len(questions),
                "performance_level": self._score_to_level(avg),
                "strongest_areas": [],
                "weakest_areas": [],
                "key_recommendations": ["Review your answers and practice with the STAR method."],
                "interview_readiness": "Summary generation encountered an error — see individual question feedback.",
                "next_steps": ["Practice more mock interviews"],
                "processing_time": round(time.time() - start, 2),
                "error": "LLM returned unparseable response",
            }
        except Exception as exc:
            logger.error("Session summary failed: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _score_to_level(score: float) -> str:
        """Map a numeric score to a human-readable performance label."""
        if score >= 9.0:
            return "exceptional"
        if score >= 7.5:
            return "strong"
        if score >= 6.0:
            return "competent"
        if score >= 4.0:
            return "developing"
        return "needs_improvement"
