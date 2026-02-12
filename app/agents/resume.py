"""
Resume Agent — LLM-powered resume analysis and improvement suggestions.

This agent uses GPT-4o-mini to provide:
- Comprehensive resume scoring and feedback
- ATS (Applicant Tracking System) compatibility assessment
- Keyword gap analysis against a target job description
- Section-by-section feedback
- Concrete, rewritten improvement suggestions
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List, Optional, Any
import json
import logging
import time

logger = logging.getLogger(__name__)


class ResumeAgent:
    """Analyzes resumes and provides structured, actionable feedback.

    Usage::

        agent = ResumeAgent()
        analysis = agent.analyze_resume(resume_text, job_description)
        improvements = agent.suggest_improvements(resume_text, job_description, analysis)
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        """Initialise the Resume Agent.

        Args:
            model: OpenAI model identifier.
            temperature: LLM sampling temperature (lower = more deterministic).
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self._analysis_prompt = self._build_analysis_prompt()
        self._improvement_prompt = self._build_improvement_prompt()
        logger.info("ResumeAgent initialised (model=%s)", model)

    # ------------------------------------------------------------------ #
    #  Prompt builders
    # ------------------------------------------------------------------ #

    def _build_analysis_prompt(self) -> ChatPromptTemplate:
        """Prompt that produces a structured JSON resume analysis."""
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert resume reviewer and career consultant with "
             "15+ years of experience in talent acquisition across technology, "
             "finance, healthcare, and other industries.\n\n"
             "Analyze the provided resume thoroughly and return your analysis "
             "as a **single JSON object** with exactly these keys:\n\n"
             "{{\n"
             '  "overall_score": <number between 1.0 and 10.0>,\n'
             '  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],\n'
             '  "weaknesses": ["<weakness 1>", "<weakness 2>", "<weakness 3>"],\n'
             '  "recommendations": [\n'
             '    "<actionable recommendation 1>",\n'
             '    "<up to 7 total recommendations>"\n'
             '  ],\n'
             '  "ats_compatibility": {{\n'
             '    "score": <number between 1.0 and 10.0>,\n'
             '    "issues": ["<ATS issue found>", "..."],\n'
             '    "suggestions": ["<ATS optimisation tip>", "..."]\n'
             '  }},\n'
             '  "keyword_analysis": {{\n'
             '    "present_keywords": ["<keyword found in resume>", "..."],\n'
             '    "missing_keywords": ["<important keyword NOT in resume>", "..."],\n'
             '    "keyword_density_notes": "<brief note on keyword usage>"\n'
             '  }},\n'
             '  "section_feedback": {{\n'
             '    "contact_info": "<feedback on contact / header section>",\n'
             '    "summary": "<feedback on professional summary or objective>",\n'
             '    "experience": "<feedback on work-experience section>",\n'
             '    "skills": "<feedback on skills section>",\n'
             '    "education": "<feedback on education section>"\n'
             '  }}\n'
             "}}\n\n"
             "Guidelines:\n"
             "- Be specific and reference actual content from the resume.\n"
             "- If a job description is provided, tailor ALL feedback to that role.\n"
             "- Score constructively: most resumes fall between 4 and 8.\n"
             "- Prioritise impact-oriented language, quantified metrics, and ATS readability.\n"
             "- Identify concrete missing keywords when a job description is given.\n\n"
             "Return ONLY the JSON object — no markdown fences, no extra text."),
            ("human",
             "Resume:\n---\n{resume_text}\n---\n\n"
             "{job_description_section}\n\n"
             "Provide your complete analysis as JSON."),
        ])

    def _build_improvement_prompt(self) -> ChatPromptTemplate:
        """Prompt that produces concrete rewrite suggestions."""
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert resume writer and career strategist. "
             "Given a resume and its prior analysis, produce concrete, "
             "rewritten improvements.\n\n"
             "Return a **single JSON object** with exactly these keys:\n\n"
             "{{\n"
             '  "improved_summary": "<a rewritten or new professional summary>",\n'
             '  "improved_bullets": [\n'
             "    {{\n"
             '      "original": "<original bullet or sentence from the resume>",\n'
             '      "improved": "<rewritten version with stronger verbs and metrics>",\n'
             '      "reasoning": "<one-sentence explanation of the improvement>"\n'
             "    }}\n"
             "  ],\n"
             '  "additional_suggestions": ["<strategic suggestion>", "..."],\n'
             '  "priority_actions": [\n'
             '    "<most impactful change 1>",\n'
             '    "<change 2>",\n'
             '    "<change 3>"\n'
             "  ]\n"
             "}}\n\n"
             "Guidelines:\n"
             "- Use powerful action verbs: Led, Architected, Delivered, Optimised, etc.\n"
             "- Add plausible quantified metrics where the original lacks them.\n"
             "- Follow BAR (Background → Action → Result) for bullet points.\n"
             "- Keep the candidate's apparent experience level realistic.\n"
             "- If a job description is provided, align improvements to that role.\n\n"
             "Return ONLY the JSON object — no markdown fences, no extra text."),
            ("human",
             "Resume:\n---\n{resume_text}\n---\n\n"
             "{job_description_section}\n\n"
             "Previous analysis summary:\n{analysis_summary}\n\n"
             "Generate specific improvements as JSON."),
        ])

    # ------------------------------------------------------------------ #
    #  JSON parsing helper
    # ------------------------------------------------------------------ #

    def _parse_llm_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM output, stripping markdown fences if present."""
        text = content.strip()

        # Strip optional ```json … ``` wrapper
        if text.startswith("```"):
            first_nl = text.find("\n")
            text = text[first_nl + 1:] if first_nl != -1 else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Attempt 1: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Attempt 2: extract outermost { … }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        logger.error("Failed to parse LLM JSON: %.300s…", text)
        raise ValueError("LLM returned invalid JSON — please retry")

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def analyze_resume(
        self,
        resume_text: str,
        job_description: str = "",
    ) -> Dict[str, Any]:
        """Analyse a resume and return structured feedback.

        Args:
            resume_text: Full plain-text content of the resume.
            job_description: Optional target job description for tailored analysis.

        Returns:
            Dict with keys: overall_score, strengths, weaknesses,
            recommendations, ats_compatibility, keyword_analysis,
            section_feedback, processing_time.
        """
        start = time.time()

        jd_section = (
            f"Target Job Description:\n---\n{job_description}\n---"
            if job_description
            else "No specific job description provided. Provide general resume feedback."
        )

        try:
            chain = self._analysis_prompt | self.llm
            result = chain.invoke({
                "resume_text": resume_text,
                "job_description_section": jd_section,
            })

            analysis = self._parse_llm_json(result.content)
            analysis["processing_time"] = round(time.time() - start, 2)

            logger.info(
                "Resume analysis completed in %.2fs — score: %s",
                analysis["processing_time"],
                analysis.get("overall_score", "N/A"),
            )
            return analysis

        except ValueError:
            # JSON parse failure — return structured fallback
            return self._error_analysis(
                "LLM returned unparseable response", time.time() - start
            )
        except Exception as exc:
            logger.error("Resume analysis failed: %s", exc, exc_info=True)
            raise

    def suggest_improvements(
        self,
        resume_text: str,
        job_description: str = "",
        analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate concrete improvement suggestions for a resume.

        Args:
            resume_text: Full plain-text content of the resume.
            job_description: Optional job description for targeted improvements.
            analysis: Previous analysis dict. When *None* a fresh analysis is
                      run automatically before generating improvements.

        Returns:
            Dict with keys: improved_summary, improved_bullets,
            additional_suggestions, priority_actions, processing_time.
        """
        start = time.time()

        # Run analysis first if not supplied
        if analysis is None:
            analysis = self.analyze_resume(resume_text, job_description)

        analysis_summary = json.dumps(
            {
                k: analysis.get(k)
                for k in ("overall_score", "strengths", "weaknesses", "recommendations")
            },
            indent=2,
        )

        jd_section = (
            f"Target Job Description:\n---\n{job_description}\n---"
            if job_description
            else "No specific job description provided."
        )

        try:
            chain = self._improvement_prompt | self.llm
            result = chain.invoke({
                "resume_text": resume_text,
                "job_description_section": jd_section,
                "analysis_summary": analysis_summary,
            })

            improvements = self._parse_llm_json(result.content)
            improvements["processing_time"] = round(time.time() - start, 2)

            logger.info(
                "Resume improvements generated in %.2fs", improvements["processing_time"]
            )
            return improvements

        except ValueError as exc:
            logger.error("Improvement suggestions parse error: %s", exc)
            return {
                "improved_summary": "",
                "improved_bullets": [],
                "additional_suggestions": analysis.get("recommendations", []),
                "priority_actions": [],
                "processing_time": round(time.time() - start, 2),
                "error": str(exc),
            }
        except Exception as exc:
            logger.error("Improvement suggestions failed: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _error_analysis(self, message: str, elapsed: float) -> Dict[str, Any]:
        """Return a well-structured fallback when analysis cannot complete."""
        return {
            "overall_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": ["Analysis could not be completed — please try again."],
            "ats_compatibility": {
                "score": 0.0,
                "issues": [message],
                "suggestions": [],
            },
            "keyword_analysis": {
                "present_keywords": [],
                "missing_keywords": [],
                "keyword_density_notes": "",
            },
            "section_feedback": {},
            "processing_time": round(elapsed, 2),
            "error": message,
        }
