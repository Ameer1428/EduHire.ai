import pandas as pd
from typing import List, Dict, Any
import json

from core.rag_engine import RAGEngine

class JobAgent:
    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine
        self.job_dataset = None
        
    def load_job_dataset(self, file_path: str):
        """Load job dataset from CSV"""
        try:
            self.job_dataset = pd.read_csv(file_path)
            print(f"✅ Loaded job dataset with {len(self.job_dataset)} entries")
        except Exception as e:
            print(f"❌ Failed to load job dataset: {e}")
    
    def process_job_query(self, user_id: str, query: str, context: Dict = None) -> Dict[str, Any]:
        """Process job-related queries"""
        if "cover letter" in query.lower() or "application" in query.lower():
            return self._handle_application_queries(user_id, query, context)
        elif "match" in query.lower() or "find jobs" in query.lower():
            return self._handle_job_matching(user_id, query, context)
        else:
            # General job query
            return self.rag_engine.query_knowledge_base(query, user_id)
    
    def find_job_matches(self, user_id: str, user_profile: Dict) -> List[Dict[str, Any]]:
        """Find job matches based on user profile"""
        if self.job_dataset is None:
            return {"error": "Job dataset not loaded"}
        
        # Convert user profile to search query
        search_query = self._profile_to_search_query(user_profile)
        
        # Use RAG to find matching jobs
        results = self.rag_engine.query_jobs(search_query, user_profile)
        
        # Rank and filter results
        ranked_jobs = self._rank_jobs(results["jobs"], user_profile)
        
        return {
            "user_id": user_id,
            "search_query": search_query,
            "matches": ranked_jobs[:10],  # Top 10 matches
            "match_metrics": {
                "total_found": len(ranked_jobs),
                "top_matches": min(10, len(ranked_jobs)),
                "match_confidence": "high" if len(ranked_jobs) > 5 else "medium"
            }
        }
    
    def generate_cover_letter(self, user_id: str, job_id: str) -> Dict[str, Any]:
        """Generate personalized cover letter for a job"""
        # Get job details
        job_details = self._get_job_details(job_id)
        if not job_details:
            return {"error": "Job not found"}
        
        # Get user profile and resume
        user_profile = self._get_user_profile(user_id)
        resume_content = self._get_resume_content(user_id)
        
        # Generate cover letter using LLM
        cover_letter_prompt = self._create_cover_letter_prompt(job_details, user_profile, resume_content)
        
        # In production, this would call the LLM
        cover_letter = self._generate_with_llm(cover_letter_prompt)
        
        return {
            "job_id": job_id,
            "job_title": job_details.get("title", "Unknown"),
            "company": job_details.get("company", "Unknown"),
            "cover_letter": cover_letter,
            "personalization_notes": [
                "Highlighted relevant skills from your resume",
                "Matched your experience to job requirements",
                "Used professional business language"
            ]
        }
    
    def _handle_application_queries(self, user_id: str, query: str, context: Dict) -> Dict[str, Any]:
        """Handle cover letter and application-related queries"""
        return {
            "type": "application_assistance",
            "user_id": user_id,
            "query": query,
            "suggestions": [
                "I can help generate personalized cover letters",
                "Provide tips for job applications",
                "Review and improve your resume",
                "Prepare for interviews"
            ],
            "next_actions": [
                "Upload your resume for better personalization",
                "Specify the job you're applying for",
                "Share the job description for tailored advice"
            ]
        }
    
    def _handle_job_matching(self, user_id: str, query: str, context: Dict) -> Dict[str, Any]:
        """Handle job matching queries"""
        user_profile = context.get("user_profile", {})
        matches = self.find_job_matches(user_id, user_profile)
        
        return {
            "type": "job_matching",
            "user_id": user_id,
            "query": query,
            "results": matches,
            "recommendations": self._get_job_search_recommendations(user_profile)
        }
    
    def _profile_to_search_query(self, user_profile: Dict) -> str:
        """Convert user profile to job search query"""
        skills = user_profile.get("skills", [])
        experience = user_profile.get("experience_level", "entry")
        location = user_profile.get("location", "")
        industry = user_profile.get("industry", "")
        
        query_parts = []
        
        if skills:
            query_parts.extend(skills[:3])  # Top 3 skills
        
        if industry:
            query_parts.append(industry)
            
        query_parts.append(f"{experience} level")
        
        return " ".join(query_parts)
    
    def _rank_jobs(self, jobs: List[Dict], user_profile: Dict) -> List[Dict]:
        """Rank jobs based on relevance to user profile"""
        # Simple ranking based on skill matching
        user_skills = set(skill.lower() for skill in user_profile.get("skills", []))
        
        for job in jobs:
            job_skills = set(skill.lower() for skill in job.get("required_skills", []))
            matching_skills = user_skills.intersection(job_skills)
            job["match_score"] = len(matching_skills) / max(len(job_skills), 1)
            job["matching_skills"] = list(matching_skills)
        
        return sorted(jobs, key=lambda x: x.get("match_score", 0), reverse=True)
    
    def _get_job_details(self, job_id: str) -> Dict:
        """Get job details from dataset"""
        if self.job_dataset is not None and "job_id" in self.job_dataset.columns:
            job_row = self.job_dataset[self.job_dataset["job_id"] == job_id]
            if not job_row.empty:
                return job_row.iloc[0].to_dict()
        return {}
    
    def _get_user_profile(self, user_id: str) -> Dict:
        """Get user profile (simplified)"""
        return {
            "skills": ["Python", "Machine Learning", "Data Analysis"],
            "experience": "2 years",
            "education": "Bachelor's in Computer Science"
        }
    
    def _get_resume_content(self, user_id: str) -> str:
        """Get user resume content (simplified)"""
        return "Experienced professional with background in software development and data science."
    
    def _create_cover_letter_prompt(self, job_details: Dict, user_profile: Dict, resume: str) -> str:
        """Create prompt for cover letter generation"""
        return f"""
        Generate a professional cover letter for the following job:
        
        Job Title: {job_details.get('title', 'N/A')}
        Company: {job_details.get('company', 'N/A')}
        Requirements: {job_details.get('requirements', 'N/A')}
        
        Applicant Profile:
        Skills: {', '.join(user_profile.get('skills', []))}
        Experience: {user_profile.get('experience', 'N/A')}
        Resume Highlights: {resume}
        
        Create a compelling, professional cover letter that:
        1. Highlights relevant skills and experience
        2. Shows enthusiasm for the specific role
        3. Is tailored to the company and position
        4. Is concise (under 400 words)
        5. Includes a call to action
        
        Cover Letter:
        """
    
    def _generate_with_llm(self, prompt: str) -> str:
        """Generate content with LLM (placeholder)"""
        # In production, this would call Azure OpenAI
        return f"[Generated cover letter based on: {prompt[:100]}...]"
    
    def _get_job_search_recommendations(self, user_profile: Dict) -> List[str]:
        """Get job search recommendations"""
        recommendations = [
            "Optimize your LinkedIn profile with relevant keywords",
            "Network with professionals in your target industry",
            "Prepare a portfolio showcasing your projects",
            "Practice common interview questions for your field"
        ]
        
        if user_profile.get("experience_level") == "entry":
            recommendations.append("Consider internships or junior positions to gain experience")
        
        if user_profile.get("skills"):
            recommendations.append(f"Highlight your {user_profile['skills'][0]} skills in applications")
            
        return recommendations