from typing import List, Dict, Any
import json
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from core.rag_engine import RAGEngine

class LearningAgent:
    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine
        self.user_profiles = {}  # In production, use database
        
    def process_learning_query(self, user_id: str, query: str, context: Dict = None) -> Dict[str, Any]:
        """Process learning-related queries"""
        # Get or create user profile
        user_profile = self._get_user_profile(user_id)
        
        # Enhance query with user context
        enhanced_query = self._enhance_query_with_context(query, user_profile, context)
        
        # Query knowledge base
        result = self.rag_engine.query_knowledge_base(enhanced_query, user_id)
        
        # Generate personalized learning path if needed
        if self._is_learning_path_query(query):
            learning_path = self._generate_learning_path(user_profile, query, result)
            result["learning_path"] = learning_path
        
        return result
    
    def get_recommendations(self, user_id: str, learning_goals: List[str]) -> Dict[str, Any]:
        """Get personalized learning recommendations"""
        user_profile = self._get_user_profile(user_id)
        
        # Build comprehensive query based on goals
        query = f"Learning resources and roadmap for: {', '.join(learning_goals)}. " \
                f"Current skills: {user_profile.get('current_skills', [])}. " \
                f"Target level: {user_profile.get('target_level', 'intermediate')}"
        
        result = self.rag_engine.query_knowledge_base(query, user_id, n_results=8)
        
        # Structure recommendations
        recommendations = self._structure_recommendations(result, learning_goals)
        
        return {
            "user_id": user_id,
            "learning_goals": learning_goals,
            "recommendations": recommendations,
            "personalized_roadmap": self._create_learning_roadmap(learning_goals, user_profile)
        }
    
    def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "current_skills": [],
                "learning_goals": [],
                "preferred_learning_style": "mixed",
                "target_level": "intermediate",
                "time_commitment": "medium"
            }
        return self.user_profiles[user_id]
    
    def _enhance_query_with_context(self, query: str, user_profile: Dict, context: Dict) -> str:
        """Enhance query with user profile and context"""
        enhanced = query
        
        if user_profile.get("current_skills"):
            enhanced += f" User skills: {', '.join(user_profile['current_skills'])}"
        
        if user_profile.get("target_level"):
            enhanced += f" Target proficiency: {user_profile['target_level']}"
            
        if context and context.get("specific_need"):
            enhanced += f" Specific need: {context['specific_need']}"
            
        return enhanced
    
    def _is_learning_path_query(self, query: str) -> bool:
        """Check if query is about learning paths"""
        path_keywords = ["learn", "study", "roadmap", "path", "course", "tutorial", "how to"]
        return any(keyword in query.lower() for keyword in path_keywords)
    
    def _generate_learning_path(self, user_profile: Dict, query: str, rag_result: Dict) -> Dict[str, Any]:
        """Generate personalized learning path"""
        # This would integrate with the LLM to create structured learning path
        # For now, return a structured format
        return {
            "query": query,
            "estimated_duration": "6-8 weeks",
            "weekly_breakdown": [
                {"week": 1, "topics": ["Fundamentals", "Basic concepts"], "resources": 3},
                {"week": 2, "topics": ["Intermediate concepts", "Practice"], "resources": 4},
                {"week": 3, "topics": ["Advanced topics", "Projects"], "resources": 5}
            ],
            "recommended_resources": rag_result.get("sources", [])[:5]
        }
    
    def _structure_recommendations(self, rag_result: Dict, learning_goals: List[str]) -> Dict[str, Any]:
        """Structure RAG results into learning recommendations"""
        return {
            "immediate_actions": [
                "Start with fundamental concepts",
                "Complete beginner tutorials",
                "Join relevant learning communities"
            ],
            "resource_categories": {
                "beginner": rag_result.get("sources", [])[:2],
                "intermediate": rag_result.get("sources", [])[2:4],
                "advanced": rag_result.get("sources", [])[4:6]
            },
            "success_metrics": [
                "Complete 2 practical projects",
                "Score 80%+ on self-assessment",
                "Build portfolio demonstrating skills"
            ]
        }
    
    def _create_learning_roadmap(self, learning_goals: List[str], user_profile: Dict) -> Dict[str, Any]:
        """Create comprehensive learning roadmap"""
        return {
            "goals": learning_goals,
            "timeline": {
                "short_term": ["Master basics", "Complete first project"],
                "medium_term": ["Build portfolio", "Gain intermediate proficiency"],
                "long_term": ["Achieve advanced skills", "Prepare for job applications"]
            },
            "milestones": [
                {"milestone": "Foundation", "duration": "2 weeks", "completion_criteria": "Basic proficiency"},
                {"milestone": "Application", "duration": "3 weeks", "completion_criteria": "Project completion"},
                {"milestone": "Mastery", "duration": "4 weeks", "completion_criteria": "Portfolio ready"}
            ]
        }