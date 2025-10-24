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
    
    def find_job_matches(self, user_id: str, user_profile: Dict, use_api: bool = True) -> Dict[str, Any]:
        """Find job matches with API integration"""
        try:
            search_query = self._profile_to_search_query(user_profile)
            
            print(f"🔍 Searching with query: '{search_query}' (API: {use_api})")
            
            # Get matches from RAG engine with API integration
            results = self.rag_engine.query_jobs(search_query, user_profile, use_api=use_api)
            
            # Format jobs for frontend
            formatted_matches = []
            for job in results.get("jobs", []):
                metadata = job.get("metadata", {})
                formatted_job = {
                    "id": job.get("id", f"job_{hash(str(metadata))}"),
                    "title": metadata.get("title", "Unknown Position"),
                    "company": metadata.get("company", "Unknown Company"),
                    "location": metadata.get("location", "Remote"),
                    "description": job.get("content", "")[:200] + "...",
                    "required_skills": metadata.get("required_skills", ""),
                    "experience_level": metadata.get("experience_level", "Not specified"),
                    "job_type": metadata.get("job_type", "Full-time"),
                    "salary": metadata.get("salary", "Not specified"),
                    "relevance_score": job.get("relevance_score", 0),
                    "skill_match": job.get("skill_match", "N/A"),
                    "apply_url": metadata.get("apply_url", ""),  # Added for API jobs
                    "source": job.get("source", "local")  # Added to identify source
                }
                formatted_matches.append(formatted_job)
            
            print(f"🎯 Formatted {len(formatted_matches)} matches ({results.get('sources', {})})")
            
            return {
                "user_id": user_id,
                "search_query": search_query,
                "matches": formatted_matches[:10],
                "match_metrics": {
                    "total_found": len(formatted_matches),
                    "top_matches": min(10, len(formatted_matches)),
                    "match_confidence": "high" if len(formatted_matches) > 5 else "medium",
                    "sources": results.get("sources", {})
                },
                "insights": results.get("insights", {}),
                "recommendations": results.get("recommendations", [])
            }
        except Exception as e:
            print(f"❌ Job matching failed: {e}")
            return {
                "error": f"Job matching failed: {str(e)}",
                "matches": [],
                "match_metrics": {"total_found": 0, "top_matches": 0, "match_confidence": "low"}
            }
    # def find_job_matches(self, user_id: str, user_profile: Dict) -> Dict[str, Any]:
    #     """Find job matches with FALLBACK search"""
    #     try:
    #         # Convert user profile to search query
    #         search_query = self._profile_to_search_query(user_profile)
            
    #         print(f"🔍 Searching with query: '{search_query}'")
            
    #         # Try different search strategies
    #         results = None
            
    #         # Strategy 1: Try exact search
    #         results = self.rag_engine.query_jobs(search_query, user_profile)
            
    #         # Strategy 2: If no results, try broader search
    #         if not results.get("jobs"):
    #             print("🔄 No results with exact search, trying broader terms...")
    #             broader_terms = [" ".join(user_profile.get("skills", ["developer"]))]
    #             for term in broader_terms:
    #                 fallback_results = self.rag_engine.query_jobs(term, user_profile)
    #                 if fallback_results.get("jobs"):
    #                     results = fallback_results
    #                     break
            
    #         # Strategy 3: If still no results, get all jobs and filter client-side
    #         if not results.get("jobs"):
    #             print("🔄 Using fallback: getting all jobs...")
    #             all_jobs = self.rag_engine.vector_store.search("job_descriptions", "", 50)
    #             if all_jobs:
    #                 # Filter and rank manually
    #                 filtered_jobs = self.rag_engine._filter_jobs_client_side(all_jobs, user_profile)
    #                 ranked_jobs = self.rag_engine._rank_jobs_by_relevance(filtered_jobs, user_profile)
                    
    #                 results = {
    #                     "jobs": ranked_jobs,
    #                     "query": search_query,
    #                     "total_found": len(ranked_jobs),
    #                     "fallback_used": True
    #                 }
            
    #         # Format jobs for frontend
    #         formatted_matches = []
    #         for job in results.get("jobs", []):
    #             metadata = job.get("metadata", {})
    #             formatted_job = {
    #                 "id": job.get("id", f"job_{hash(str(metadata))}"),
    #                 "title": metadata.get("title", "Unknown Position"),
    #                 "company": metadata.get("company", "Unknown Company"),
    #                 "location": metadata.get("location", "Remote"),
    #                 "description": job.get("content", "")[:200] + "...",
    #                 "required_skills": metadata.get("required_skills", ""),
    #                 "experience_level": metadata.get("experience_level", "Not specified"),
    #                 "job_type": metadata.get("job_type", "Full-time"),
    #                 "salary": metadata.get("salary", "Not specified"),
    #                 "relevance_score": job.get("relevance_score", 0),
    #                 "skill_match": job.get("skill_match", "N/A")
    #             }
    #             formatted_matches.append(formatted_job)
            
    #         print(f"🎯 Formatted {len(formatted_matches)} matches for frontend")
            
    #         return {
    #             "user_id": user_id,
    #             "search_query": search_query,
    #             "matches": formatted_matches[:10],
    #             "match_metrics": {
    #                 "total_found": len(formatted_matches),
    #                 "top_matches": min(10, len(formatted_matches)),
    #                 "match_confidence": "high" if len(formatted_matches) > 5 else "medium"
    #             },
    #             "insights": results.get("insights", {}),
    #             "recommendations": results.get("recommendations", [])
    #         }
            
    #     except Exception as e:
    #         print(f"❌ Job matching failed: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return {
    #             "error": f"Job matching failed: {str(e)}",
    #             "matches": [],
    #             "match_metrics": {"total_found": 0, "top_matches": 0, "match_confidence": "low"}
    #         }
    
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
        """Convert user profile to job search query - SIMPLIFIED FOR API"""
        skills = user_profile.get("skills", [])
        experience = user_profile.get("experience_level", "")
        location = user_profile.get("location", "")
        keywords = user_profile.get("keywords", "")
        
        query_parts = []
        
        # Priority 1: Use keywords if provided (most specific)
        if keywords and keywords.strip():
            query_parts.append(keywords.strip())
        
        # Priority 2: Add 1-2 main skills (keep it simple for API)
        if skills:
            if isinstance(skills, list):
                # Use only the first 1-2 skills to avoid overly specific queries
                main_skills = skills[:2]
                query_parts.extend(main_skills)
            else:
                query_parts.append(str(skills))
        
        # Priority 3: Add experience level only if not already in keywords
        if experience and experience.lower() in ["entry", "junior"]:
            if not any(exp_term in " ".join(query_parts).lower() for exp_term in ["entry", "junior", "fresher"]):
                query_parts.append("entry level")
        elif experience and experience.lower() == "senior":
            if not any(exp_term in " ".join(query_parts).lower() for exp_term in ["senior", "lead", "principal"]):
                query_parts.append("senior")
        
        # If no specific query, use broader terms that work well with API
        if not query_parts:
            query_parts = ["developer", "software engineer", "technology"]
        
        # Limit query length for API compatibility (max 3-4 terms)
        final_query = " ".join(query_parts[:3])
        print(f"🎯 Generated API search query: '{final_query}'")
        return final_query
        
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