"""
Memory Agent — Conversation persistence and user profile management.

Manages user conversation history, profiles, and contextual memory across sessions.
Provides LLM-powered conversation summarization and profile updates based on 
interaction patterns.

Usage::

    from app.agents.memory import MemoryAgent

    agent = MemoryAgent()
    
    # Load context for other agents
    context = agent.load_user_context(user_id=1)
    
    # Save conversation with summarization
    agent.save_conversation_with_analysis(
        user_id=1, session_id="abc123", 
        message="I want to practice Python interviews",
        agent_response="Let's start with data structures..."
    )
    
    # Update profile based on conversation insights
    agent.update_profile_from_conversation(user_id=1, session_id="abc123")
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Union

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.tools.database import DatabaseManager, get_db

logger = logging.getLogger(__name__)


class MemoryAgent:
    """Intelligent conversation and profile management agent."""

    def __init__(
        self, 
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        db: Optional[DatabaseManager] = None
    ):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.db = db or get_db()
        
        # ---- conversation summarization prompt -------------------------
        self.summarization_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are an expert conversation analyst for a career coaching system. "
                    "Create a concise summary of the conversation that captures:\n"
                    "1. User's main goals and intent\n"
                    "2. Key information shared (skills, experience, preferences)\n"
                    "3. Agent guidance provided\n"
                    "4. Important decisions or insights\n"
                    "5. Follow-up needs or next steps\n\n"
                    "Keep summaries under 200 words and focus on actionable insights.\n\n"
                    "Return as JSON with this exact structure:\n"
                    "{{\n"
                    '  "summary": "<concise conversation summary>",\n'
                    '  "key_insights": ["insight 1", "insight 2", "insight 3"],\n'
                    '  "user_goals": ["goal 1", "goal 2"],\n'
                    '  "next_actions": ["action 1", "action 2"]\n'
                    "}}\n\n"
                    "Return ONLY valid JSON — no markdown fences."
                )
            ),
            (
                "human",
                (
                    "Conversation to summarize:\n"
                    "USER: {user_message}\n"
                    "AGENT ({agent_type}): {agent_response}\n\n"
                    "Previous context: {previous_context}"
                )
            )
        ])
        
        # ---- profile extraction prompt ------------------------------
        self.profile_extraction_prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                (
                    "You are a profile analyst for a career coaching system. "
                    "Extract and update user profile information based on conversation content.\n\n"
                    "Focus on extracting:\n"
                    "- Technical skills mentioned\n"
                    "- Career level and experience\n"
                    "- Target roles and companies\n"
                    "- Industry preferences\n"
                    "- Location preferences\n"
                    "- Strengths and areas for improvement\n"
                    "- Career goals and timeline\n\n"
                    "Return as JSON with this structure:\n"
                    "{{\n"
                    '  "skills": {"technical": ["skill1", "skill2"], "soft": ["skill1", "skill2"]},\n'
                    '  "experience_level": "junior|mid|senior|lead",\n'
                    '  "target_roles": ["role1", "role2"],\n'
                    '  "industries": ["industry1", "industry2"],\n'
                    '  "location_preferences": ["location1", "location2"],\n'
                    '  "career_goals": ["goal1", "goal2"],\n'
                    '  "strengths": ["strength1", "strength2"],\n'
                    '  "improvement_areas": ["area1", "area2"]\n'
                    "}}\n\n"
                    "Only include information explicitly mentioned or strongly implied. "
                    "Return ONLY valid JSON — no markdown fences."
                )
            ),
            (
                "human",
                (
                    "Current user profile:\n"
                    "{current_profile}\n\n"
                    "New conversation to analyze:\n"
                    "USER: {user_message}\n"
                    "AGENT: {agent_response}\n\n"
                    "Extract new profile information or updates from this conversation."
                )
            )
        ])
        
        # ---- context formatting prompt -----------------------------
        self.context_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are a context curator for a career coaching system. "
                    "Given a user's conversation history and profile, create a concise "
                    "context summary for other agents to use.\n\n"
                    "Focus on information relevant to the current interaction type: {interaction_type}\n\n"
                    "Return as JSON:\n"
                    "{{\n"
                    '  "user_background": "<brief background summary>",\n'
                    '  "relevant_history": "<relevant conversation context>",\n'
                    '  "preferences": "<user preferences and goals>",\n'
                    '  "context_notes": "<additional context for the agent>"\n'
                    "}}\n\n"
                    "Keep each field under 100 words. Return ONLY valid JSON."
                )
            ),
            (
                "human",
                (
                    "User profile: {user_profile}\n\n"
                    "Recent conversation history:\n"
                    "{conversation_history}\n\n"
                    "Current interaction type: {interaction_type}"
                )
            )
        ])

        # ---- chains --------------------------------------------------
        self.summarization_chain = self.summarization_prompt | self.llm
        self.profile_chain = self.profile_extraction_prompt | self.llm
        self.context_chain = self.context_prompt | self.llm

    # ------------------------------------------------------------------ #
    #  Core public methods
    # ------------------------------------------------------------------ #

    def load_user_context(
        self, 
        user_id: int,
        interaction_type: str = "general",
        history_limit: int = 5
    ) -> Dict[str, Any]:
        """Load comprehensive user context for other agents.
        
        Returns user profile, recent conversation history, and formatted
        context suitable for injection into other agents' prompts.
        """
        start_time = time.time()
        
        try:
            # Get user profile
            user_data = self.db.get_user(user_id)
            profile = user_data.get("profile_data", {}) if user_data else {}
            preferences = user_data.get("preferences", {}) if user_data else {}
            
            # Get recent conversation history
            conversations = self.db.get_conversation_history(
                user_id=user_id, limit=history_limit
            )
            
            # Format conversation history
            history_text = self._format_conversation_history(conversations)
            
            # Generate contextual summary for agents
            context_summary = self._generate_context_summary(
                profile, history_text, interaction_type
            )
            
            return {
                "user_id": user_id,
                "profile": profile,
                "preferences": preferences,
                "recent_conversations": conversations,
                "context_summary": context_summary,
                "history_count": len(conversations),
                "processing_time": round(time.time() - start_time, 3)
            }
            
        except Exception as e:
            logger.error("Failed to load user context: %s", e)
            return {
                "user_id": user_id,
                "profile": {},
                "preferences": {},
                "recent_conversations": [],
                "context_summary": {"user_background": "", "relevant_history": "", 
                                   "preferences": "", "context_notes": ""},
                "history_count": 0,
                "processing_time": round(time.time() - start_time, 3)
            }
    
    def save_conversation_with_analysis(
        self, 
        user_id: int,
        session_id: str,
        user_message: str,
        agent_response: str,
        agent_type: str = "general",
        intent: str = "unknown"
    ) -> Dict[str, Any]:
        """Save conversation and generate intelligent summary/analysis."""
        start_time = time.time()
        
        try:
            # Generate conversation summary
            summary_result = self._summarize_conversation(
                user_message, agent_response, agent_type, user_id
            )
            
            # Save conversation to database
            conversation_id = self.db.save_conversation(
                user_id=user_id,
                session_id=session_id,
                message=user_message,
                response=agent_response,
                intent=intent,
                agent_used=agent_type,
                metadata=summary_result
            )
            
            return {
                "conversation_id": conversation_id,
                "summary": summary_result,
                "saved": conversation_id is not None,
                "processing_time": round(time.time() - start_time, 3)
            }
            
        except Exception as e:
            logger.error("Failed to save conversation with analysis: %s", e)
            # Fallback: save basic conversation without analysis
            try:
                conversation_id = self.db.save_conversation(
                    user_id=user_id,
                    session_id=session_id,
                    message=user_message,
                    response=agent_response,
                    intent=intent,
                    agent_used=agent_type
                )
                return {
                    "conversation_id": conversation_id,
                    "summary": None,
                    "saved": conversation_id is not None,
                    "processing_time": round(time.time() - start_time, 3)
                }
            except Exception as fallback_err:
                logger.error("Failed basic conversation save: %s", fallback_err)
                return {
                    "conversation_id": None,
                    "summary": None,
                    "saved": False,
                    "processing_time": round(time.time() - start_time, 3)
                }

    def update_profile_from_conversation(
        self,
        user_id: int,
        user_message: str,
        agent_response: str
    ) -> Dict[str, Any]:
        """Extract insights from conversation and update user profile."""
        start_time = time.time()
        
        try:
            # Get current profile
            user_data = self.db.get_user(user_id) or {}
            current_profile = user_data.get("profile_data", {})
            
            # Extract profile updates from conversation
            profile_updates = self._extract_profile_updates(
                current_profile, user_message, agent_response
            )
            
            if profile_updates:
                # Merge with existing profile
                merged_profile = self._merge_profile_data(current_profile, profile_updates)
                
                # Update in database
                success = self.db.update_user_profile(user_id, merged_profile)
                
                return {
                    "updated": success,
                    "new_insights": profile_updates,
                    "full_profile": merged_profile if success else current_profile,
                    "processing_time": round(time.time() - start_time, 3)
                }
            else:
                return {
                    "updated": False,
                    "new_insights": {},
                    "full_profile": current_profile,
                    "processing_time": round(time.time() - start_time, 3)
                }
                
        except Exception as e:
            logger.error("Failed to update profile from conversation: %s", e)
            return {
                "updated": False,
                "new_insights": {},
                "full_profile": {},
                "processing_time": round(time.time() - start_time, 3)
            }

    def get_conversation_insights(
        self,
        user_id: int,
        session_id: Optional[str] = None,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """Get analytical insights from recent conversations."""
        try:
            # Get conversations
            conversations = self.db.get_conversation_history(
                user_id=user_id, 
                session_id=session_id,
                limit=20
            )
            
            if not conversations:
                return {
                    "insights": [],
                    "patterns": [],
                    "recommendations": [],
                    "conversation_count": 0
                }
            
            # Extract insights from metadata
            insights = []
            patterns = []
            agent_usage = {}
            
            for conv in conversations:
                metadata = conv.get("metadata", {})
                if isinstance(metadata, dict):
                    if "key_insights" in metadata:
                        insights.extend(metadata["key_insights"])
                    
                    agent = conv.get("agent_used", "unknown")
                    agent_usage[agent] = agent_usage.get(agent, 0) + 1
            
            # Identify patterns
            if agent_usage:
                most_used_agent = max(agent_usage, key=agent_usage.get)
                patterns.append(f"Primary focus on {most_used_agent} coaching")
            
            # Generate recommendations
            recommendations = self._generate_insights_recommendations(
                insights, patterns, agent_usage
            )
            
            return {
                "insights": list(set(insights))[:10],  # Deduplicate and limit
                "patterns": patterns,
                "recommendations": recommendations,
                "conversation_count": len(conversations),
                "agent_usage": agent_usage
            }
            
        except Exception as e:
            logger.error("Failed to get conversation insights: %s", e)
            return {
                "insights": [],
                "patterns": [],
                "recommendations": [],
                "conversation_count": 0
            }

    # ------------------------------------------------------------------ #
    #  Internal helper methods
    # ------------------------------------------------------------------ #

    def _summarize_conversation(
        self, 
        user_message: str, 
        agent_response: str, 
        agent_type: str,
        user_id: int
    ) -> Optional[Dict[str, Any]]:
        """Generate LLM-powered conversation summary."""
        try:
            # Get previous context for better summarization
            recent_history = self.db.get_conversation_history(user_id, limit=3)
            previous_context = self._format_previous_context(recent_history)
            
            response = self.summarization_chain.invoke({
                "user_message": user_message,
                "agent_response": agent_response[:1000],  # Truncate long responses
                "agent_type": agent_type,
                "previous_context": previous_context
            })
            
            return self._parse_llm_json(response.content)
            
        except Exception as e:
            logger.error("Conversation summarization failed: %s", e)
            return None

    def _extract_profile_updates(
        self, 
        current_profile: Dict, 
        user_message: str, 
        agent_response: str
    ) -> Optional[Dict[str, Any]]:
        """Extract profile information from conversation using LLM."""
        try:
            response = self.profile_chain.invoke({
                "current_profile": json.dumps(current_profile, default=str),
                "user_message": user_message,
                "agent_response": agent_response[:800]  # Truncate for token efficiency
            })
            
            return self._parse_llm_json(response.content)
            
        except Exception as e:
            logger.error("Profile extraction failed: %s", e)
            return None

    def _generate_context_summary(
        self, 
        profile: Dict, 
        history_text: str, 
        interaction_type: str
    ) -> Dict[str, str]:
        """Generate formatted context summary for other agents."""
        try:
            response = self.context_chain.invoke({
                "user_profile": json.dumps(profile, default=str),
                "conversation_history": history_text[:1500],  # Limit context size
                "interaction_type": interaction_type
            })
            
            parsed = self._parse_llm_json(response.content)
            if parsed:
                return parsed
            else:
                # Fallback to basic context
                return self._basic_context_summary(profile, history_text)
                
        except Exception as e:
            logger.error("Context generation failed: %s", e)
            return self._basic_context_summary(profile, history_text)

    def _format_conversation_history(self, conversations: List[Dict]) -> str:
        """Format conversation list into readable text."""
        if not conversations:
            return "No previous conversations."
        
        formatted_lines = []
        for conv in conversations[-5:]:  # Last 5 conversations
            timestamp = conv.get("created_at", "")
            intent = conv.get("intent", "")
            message = conv.get("message", "")[:100]  # Truncate
            formatted_lines.append(f"[{timestamp}] ({intent}): {message}...")
        
        return "\n".join(formatted_lines)

    def _format_previous_context(self, conversations: List[Dict]) -> str:
        """Format recent conversations for context injection."""
        if not conversations:
            return "No previous context."
        
        context_lines = []
        for conv in conversations[-2:]:  # Last 2 conversations
            message = conv.get("message", "")[:150]
            context_lines.append(f"Previous: {message}")
        
        return " | ".join(context_lines)

    def _basic_context_summary(self, profile: Dict, history_text: str) -> Dict[str, str]:
        """Fallback context summary without LLM."""
        skills = profile.get("skills", {})
        technical_skills = skills.get("technical", [])
        experience_level = profile.get("experience_level", "unknown")
        
        return {
            "user_background": f"Experience level: {experience_level}. Skills: {', '.join(technical_skills[:3])}",
            "relevant_history": "Recent activity available" if history_text else "No recent activity",
            "preferences": json.dumps(profile.get("career_goals", []))[:100],
            "context_notes": "Profile data available for personalization"
        }

    def _merge_profile_data(self, current: Dict, updates: Dict) -> Dict:
        """Intelligently merge profile updates with existing data."""
        merged = current.copy()
        
        for key, value in updates.items():
            if key in merged:
                if isinstance(merged[key], dict) and isinstance(value, dict):
                    # Merge nested dictionaries
                    merged[key] = {**merged[key], **value}
                elif isinstance(merged[key], list) and isinstance(value, list):
                    # Merge lists, removing duplicates
                    merged[key] = list(set(merged[key] + value))
                else:
                    # Replace scalar values
                    merged[key] = value
            else:
                merged[key] = value
        
        return merged

    def _generate_insights_recommendations(
        self, 
        insights: List[str], 
        patterns: List[str], 
        agent_usage: Dict[str, int]
    ) -> List[str]:
        """Generate actionable recommendations based on conversation insights."""
        recommendations = []
        
        if not insights:
            recommendations.append("Continue engaging with the coaching system to build your profile")
            return recommendations
        
        # Agent usage recommendations
        if agent_usage:
            resume_count = agent_usage.get("resume", 0)
            interview_count = agent_usage.get("interview", 0)
            knowledge_count = agent_usage.get("knowledge", 0)
            
            if resume_count == 0:
                recommendations.append("Consider getting your resume reviewed for optimization")
            if interview_count == 0:
                recommendations.append("Practice mock interviews to build confidence")
            if knowledge_count > 3:
                recommendations.append("Apply the knowledge you've gained to real applications")
        
        # Pattern-based recommendations
        if len(insights) > 5:
            recommendations.append("You're making good progress! Consider scheduling regular practice sessions")
        
        return recommendations[:5]  # Limit recommendations

    @staticmethod
    def _parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM output, handling markdown fences and errors."""
        try:
            # Strip markdown fences
            cleaned = re.sub(r"```(?:json)?\s*", "", text)
            cleaned = cleaned.strip().rstrip("`")
            
            # Direct parse attempt
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
            
            # Find outermost braces
            brace_start = cleaned.find("{")
            brace_end = cleaned.rfind("}")
            if brace_start != -1 and brace_end > brace_start:
                return json.loads(cleaned[brace_start:brace_end + 1])
            
        except Exception as e:
            logger.warning("JSON parsing failed: %s", e)
        
        return None