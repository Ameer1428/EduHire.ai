import os
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class JobAPIService:
    def __init__(self):
        self.api_key = os.getenv("JOB_API_KEY")
        self.base_url = "https://jsearch.p.rapidapi.com/search"
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "jsearch.p.rapidapi.com"
        }
    
    def search_jobs(self, query: str, location: str = "", page: int = 1, num_pages: int = 1) -> Dict[str, Any]:
        """Search jobs using JSearch API - IMPROVED QUERY HANDLING"""
        try:
            if not self.api_key:
                print("❌ JOB_API_KEY not found in environment variables")
                return {"error": "API key not configured", "data": []}
            
            # Clean and optimize the query for API
            clean_query = self._optimize_query_for_api(query)
            clean_location = self._clean_location(location)
            
            querystring = {
                "query": clean_query,
                "page": str(page),
                "num_pages": str(num_pages),
                "date_posted": "all"  # Remove country restriction for broader results
            }
            
            # Only add location if it's specific (not "Remote")
            if clean_location and clean_location.lower() not in ["remote", "any", "flexible"]:
                querystring["query"] = f"{clean_query} in {clean_location}"
            
            print(f"🔍 Searching JSearch API: '{querystring['query']}'")
            
            response = requests.get(self.base_url, headers=self.headers, params=querystring)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the response to match our internal structure
            formatted_jobs = self._format_api_jobs(data.get("data", []))
            
            print(f"✅ Found {len(formatted_jobs)} jobs from JSearch API")
            return {
                "success": True,
                "jobs": formatted_jobs,
                "total": len(formatted_jobs),
                "source": "jsearch_api",
                "query_used": querystring['query']
            }
            
        except requests.exceptions.RequestException as e:
            print(f"❌ JSearch API request failed: {e}")
            return {"error": str(e), "data": []}
        except Exception as e:
            print(f"❌ JSearch API error: {e}")
            return {"error": str(e), "data": []}

    def _optimize_query_for_api(self, query: str) -> str:
        """Optimize query for better API results"""
        # Remove common stop words and limit length
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = query.split()
        
        # Keep only meaningful words and limit to 4 words max
        meaningful_words = [word for word in words if word.lower() not in stop_words]
        optimized_query = " ".join(meaningful_words[:4])
        
        return optimized_query if optimized_query else "developer"

    def _clean_location(self, location: str) -> str:
        """Clean location string for API"""
        if not location:
            return ""
        
        # Remove common generic terms
        location_lower = location.lower()
        if any(term in location_lower for term in ['remote', 'any', 'flexible', 'hybrid']):
            return ""  # Don't specify location for remote preferences
        
        # Extract city name if it's in a longer string
        cities = ['hyderabad', 'bangalore', 'pune', 'chennai', 'mumbai', 'delhi', 'gurgaon']
        for city in cities:
            if city in location_lower:
                return city.capitalize()
        
        return location
    
    def _format_api_jobs(self, api_jobs: List[Dict]) -> List[Dict]:
        """Format JSearch API jobs to match our internal structure"""
        formatted_jobs = []
        
        for job in api_jobs:
            try:
                # Extract job details with fallbacks
                job_title = job.get("job_title", "Unknown Position")
                company = job.get("employer_name", "Unknown Company")
                location = job.get("job_city") or job.get("job_country") or "Remote"
                
                # Create job content similar to our CSV format
                job_content = f"""
                    Title: {job_title}
                    Company: {company}
                    Location: {location}
                    Description: {job.get('job_description', 'No description available')}
                    Required Skills: {self._extract_skills_from_description(job.get('job_description', ''))}
                    Experience Level: {self._infer_experience_level(job.get('job_description', ''))}
                    Job Type: {job.get('job_employment_type', 'Full-time')}
                    Salary: {job.get('job_salary', 'Not specified')}
                    Posted Date: {job.get('job_posted_at_datetime_utc', 'Not specified')}
                    Apply URL: {job.get('job_apply_link', '')}
                    """
                
                # Create metadata
                job_metadata = {
                    "title": job_title,
                    "company": company,
                    "location": location,
                    "description": job.get('job_description', '')[:500],
                    "required_skills": self._extract_skills_from_description(job.get('job_description', '')),
                    "experience_level": self._infer_experience_level(job.get('job_description', '')),
                    "job_type": job.get('job_employment_type', 'Full-time'),
                    "salary": job.get('job_salary', 'Not specified'),
                    "min_salary": self._extract_min_salary(job.get('job_salary', '')),
                    "posted_date": job.get('job_posted_at_datetime_utc', ''),
                    "apply_url": job.get('job_apply_link', ''),
                    "job_id": job.get('job_id', ''),
                    "user_id": "system",
                    "doc_type": "job",
                    "source": "jsearch_api",
                    "external_id": job.get('job_id', '')
                }
                
                formatted_jobs.append({
                    "content": job_content,
                    "metadata": job_metadata,
                    "relevance_score": 0.7,  # Default score for API jobs
                    "source": "api"
                })
                
            except Exception as e:
                print(f"⚠️ Failed to format API job: {e}")
                continue
        
        return formatted_jobs
    
    def _extract_skills_from_description(self, description: str) -> str:
        """Extract skills from job description"""
        skills_keywords = [
            'Python', 'Java', 'JavaScript', 'TypeScript', 'React', 'Node.js', 'AWS', 'Azure', 'GCP',
            'Machine Learning', 'AI', 'LLM', 'GenAI', 'RAG', 'LangChain', 'OpenAI', 'Docker', 'Kubernetes',
            'SQL', 'NoSQL', 'MongoDB', 'PostgreSQL', 'FastAPI', 'Flask', 'Django', 'REST API', 'Microservices',
            'TensorFlow', 'PyTorch', 'scikit-learn', 'pandas', 'numpy', 'Git', 'Linux', 'Agile'
        ]
        
        found_skills = []
        desc_lower = str(description).lower()
        
        for skill in skills_keywords:
            if skill.lower() in desc_lower:
                found_skills.append(skill)
        
        return ", ".join(found_skills[:8]) if found_skills else "Not specified"
    
    def _infer_experience_level(self, description: str) -> str:
        """Infer experience level from description"""
        desc_lower = str(description).lower()
        
        if any(term in desc_lower for term in ['senior', 'lead', 'principal', '5+ years', '8+ years']):
            return "Senior"
        elif any(term in desc_lower for term in ['mid-level', 'intermediate', '3+ years', '2+ years']):
            return "Mid-level"
        elif any(term in desc_lower for term in ['junior', 'entry level', 'fresher', '0-2 years']):
            return "Entry-level"
        else:
            return "Not specified"
    
    def _extract_min_salary(self, salary_str: str) -> int:
        """Extract minimum salary from salary string"""
        try:
            if not salary_str or salary_str == "Not specified":
                return 0
            
            # Handle different salary formats
            import re
            numbers = re.findall(r'\d+', str(salary_str).replace(',', ''))
            if numbers:
                return int(numbers[0])
        except:
            pass
        
        return 0