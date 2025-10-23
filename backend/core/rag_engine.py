import os
import re
import uuid
import pandas as pd
from typing import List, Dict, Any
from langchain.schema import BaseRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from datetime import datetime
from .vector_store import VectorStore

class RAGEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = self._setup_llm()
        self.document_processor = DocumentProcessor(vector_store) 
        
    def _setup_llm(self):
        """Setup Azure OpenAI LLM - FIXED for your specific model"""
        try:
            print("🔧 Initializing Azure OpenAI for RAG...")
            
            # Fixed configuration - move max_completion_tokens to model_kwargs
            llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://openai-25.openai.azure.com/"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-mini"),
                temperature=1,
                model_kwargs={
                    "max_completion_tokens": 3000
                }
            )
            
            # Test the connection
            print("🚀 Testing Azure OpenAI connection...")
            test_response = llm.invoke("Hello, test connection")
            print(f"✅ Azure OpenAI connected: Received {len(str(test_response))} chars")
            
            return llm
            
        except Exception as e:
            print(f"❌ Failed to initialize Azure OpenAI: {e}")
            raise e
    
    def process_documents(self, documents: List[Any], user_id: str = None) -> Dict[str, Any]:
        """Public method to process uploaded documents"""
        return self.document_processor.process_uploaded_documents(documents, user_id)

    def query_jobs(self, query: str, user_profile: Dict = None, n_results: int = 10) -> Dict[str, Any]:
        """Enhanced job query with client-side filtering - MORE RELIABLE"""
        try:
            # Search without filters first (get more results to filter)
            enhanced_query = self._enhance_job_query(query, user_profile)
            results = self.vector_store.search("job_descriptions", enhanced_query, n_results * 3)  # Get more results
            
            # Filter results client-side based on user profile
            filtered_results = self._filter_jobs_client_side(results, user_profile)
            
            # Score and rank jobs based on relevance to user profile
            ranked_results = self._rank_jobs_by_relevance(filtered_results, user_profile)
            
            # Extract insights from job results
            insights = self._extract_job_insights(ranked_results)
            
            return {
                "jobs": ranked_results[:n_results],
                "query": query,
                "filters_applied": user_profile,
                "insights": insights,
                "total_found": len(ranked_results),
                "recommendations": self._generate_job_recommendations(ranked_results, user_profile)
            }
            
        except Exception as e:
            print(f"❌ Job query failed: {e}")
            return {
                "jobs": [],
                "error": f"Job search failed: {str(e)}"
            }

    def _filter_jobs_client_side(self, jobs: List[Dict], user_profile: Dict) -> List[Dict]:
        """Filter jobs client-side based on user profile"""
        if not user_profile:
            return jobs
        
        filtered_jobs = []
        
        for job in jobs:
            match = True
            metadata = job.get("metadata", {})
            
            # Location filter
            if user_profile.get("location"):
                user_locations = [loc.strip().lower() for loc in user_profile["location"].split(',')]
                job_location = str(metadata.get("location", "")).lower()
                
                location_match = False
                for user_loc in user_locations:
                    if user_loc in job_location or job_location in user_loc:
                        location_match = True
                        break
                
                if not location_match:
                    match = False
            
            # Experience level filter
            if user_profile.get("experience_level") and match:
                user_exp = user_profile["experience_level"].lower()
                job_exp = str(metadata.get("experience_level", "")).lower()
                
                if user_exp and job_exp and user_exp not in job_exp:
                    match = False
            
            if match:
                filtered_jobs.append(job)
        
        print(f"🔍 Filtered {len(jobs)} jobs down to {len(filtered_jobs)} based on profile")
        return filtered_jobs

    def _build_job_filters(self, user_profile: Dict) -> Dict:
        """Build SIMPLIFIED job filters - FIXED ChromaDB syntax"""
        if not user_profile:
            return None
        
        filters = {}
        
        # Build individual filters
        if user_profile.get("location"):
            # Handle multiple locations (comma-separated)
            locations = [loc.strip() for loc in user_profile["location"].split(',')]
            if len(locations) == 1:
                filters["location"] = {"$eq": locations[0]}
            else:
                filters["location"] = {"$in": locations}
        
        if user_profile.get("experience_level"):
            filters["experience_level"] = {"$eq": user_profile["experience_level"]}
        
        # Return the filter object directly (ChromaDB handles multiple conditions)
        return filters if filters else None

    def query_knowledge_base(self, query: str, user_id: str = None, 
                           n_results: int = 5) -> Dict[str, Any]:
        """Query knowledge base with RAG - FIXED FILTERS"""
        try:
            # Build proper ChromaDB filter syntax
            filters = None
            if user_id:
                filters = {"user_id": {"$eq": user_id}}
            
            results = self.vector_store.search("knowledge_base", query, n_results, filters)
            
            # Build context from retrieved documents
            context = "\n\n".join([r["content"] for r in results])
            
            # Create enhanced prompt
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are an expert career and learning advisor. Use the following context to answer the user's question.

                Context from knowledge base:
                {context}

                User Question: {question}

                Provide a comprehensive, helpful answer based on the context. If the context doesn't contain relevant information, use your general knowledge but indicate this.

                Answer:"""
            )
            
            # Generate response
            prompt = prompt_template.format(context=context, question=query)
            response = self.llm.invoke(prompt)
            
            return {
                "answer": response.content,
                "sources": results,
                "context_used": context[:500] + "..." if len(context) > 500 else context
            }
            
        except Exception as e:
            print(f"❌ RAG query failed: {e}")
            return {
                "error": f"Knowledge base query failed: {str(e)}",
                "answer": "I apologize, but I'm having trouble accessing the knowledge base right now.",
                "sources": []
            }

    def _enhance_job_query(self, query: str, user_profile: Dict) -> str:
        """Enhance job search query with user context"""
        enhanced_terms = [query]
        
        if user_profile:
            # Add skills to query if relevant
            if user_profile.get("skills") and len(query.split()) < 5:
                enhanced_terms.extend(user_profile["skills"][:2])
            
            # Add career interests
            if user_profile.get("career_interests"):
                enhanced_terms.extend(user_profile["career_interests"][:1])
        
        return " ".join(enhanced_terms)

    def _rank_jobs_by_relevance(self, jobs: List[Dict], user_profile: Dict) -> List[Dict]:
        """Rank jobs based on relevance to user profile - FIXED metadata access"""
        if not user_profile:
            return jobs
        
        scored_jobs = []
        user_skills = set(user_profile.get("skills", []))
        
        for job in jobs:
            score = 0
            metadata = job.get("metadata", {})
            
            # Skill matching (40% weight) - FIXED for string format
            job_skills = set()
            skills_data = metadata.get("required_skills", "")
            if isinstance(skills_data, str) and skills_data:
                # Convert comma-separated string back to set
                job_skills = set([s.strip() for s in skills_data.split(',')])
            elif isinstance(skills_data, list):
                job_skills = set(skills_data)
            
            if job_skills:
                skill_match_ratio = len(user_skills.intersection(job_skills)) / len(job_skills)
                score += skill_match_ratio * 40
            
            # Experience level matching (20% weight)
            user_exp = user_profile.get("experience_level", "").lower()
            job_exp = str(metadata.get("experience_level", "")).lower()
            if user_exp and job_exp and user_exp in job_exp:
                score += 20
            
            # Location matching (20% weight)
            user_location = user_profile.get("location", "").lower()
            job_location = str(metadata.get("location", "")).lower()
            if user_location and job_location:
                if user_location in job_location or job_location in user_location:
                    score += 20
                elif "remote" in job_location and user_profile.get("prefers_remote", False):
                    score += 15
            
            # Salary expectations (10% weight)
            user_min_salary = user_profile.get("min_salary", 0)
            job_min_salary = metadata.get("min_salary", 0)
            if job_min_salary and job_min_salary >= user_min_salary:
                score += 10
            
            # Job type preference (10% weight)
            preferred_types = user_profile.get("preferred_job_types", [])
            job_type = str(metadata.get("job_type", "")).lower()
            if job_type and preferred_types:
                if any(pref.lower() in job_type for pref in preferred_types):
                    score += 10
            
            job["relevance_score"] = min(100, round(score, 1))
            job["skill_match"] = f"{len(user_skills.intersection(job_skills))}/{len(job_skills)}" if job_skills else "N/A"
            scored_jobs.append(job)
        
        return sorted(scored_jobs, key=lambda x: x["relevance_score"], reverse=True)

    def _extract_job_insights(self, jobs: List[Dict]) -> Dict:
        """Extract insights from job search results - FIXED metadata access"""
        if not jobs:
            return {}
        
        insights = {
            "top_skills_demanded": [],
            "salary_ranges": {},
            "companies_hiring": [],
            "experience_level_distribution": {}
        }
        
        # Analyze top skills - FIXED for string format
        all_skills = []
        for job in jobs:
            metadata = job.get("metadata", {})
            skills_data = metadata.get("required_skills", "")
            if isinstance(skills_data, str) and skills_data:
                # Convert comma-separated string to list
                skills_list = [s.strip() for s in skills_data.split(',')]
                all_skills.extend(skills_list)
            elif isinstance(skills_data, list):
                all_skills.extend(skills_data)
        
        from collections import Counter
        skill_counts = Counter(all_skills)
        insights["top_skills_demanded"] = skill_counts.most_common(10)
        
        # Analyze companies
        companies = [str(job.get("metadata", {}).get("company", "")) for job in jobs if job.get("metadata", {}).get("company")]
        company_counts = Counter(companies)
        insights["companies_hiring"] = company_counts.most_common(5)
        
        # Analyze experience levels
        experience_levels = [str(job.get("metadata", {}).get("experience_level", "")) for job in jobs if job.get("metadata", {}).get("experience_level")]
        exp_counts = Counter(experience_levels)
        insights["experience_level_distribution"] = dict(exp_counts.most_common())
        
        return insights

    def _generate_job_recommendations(self, jobs: List[Dict], user_profile: Dict) -> List[str]:
        """Generate personalized job recommendations - FIXED for string skills"""
        recommendations = []
        
        if not user_profile or not jobs:
            return recommendations
        
        # Skill gap analysis - FIXED for string format
        user_skills = set(user_profile.get("skills", []))
        top_job_skills = set()
        
        for job in jobs[:5]:
            skills_data = job.get("required_skills", "")
            if isinstance(skills_data, str) and skills_data:
                skills_list = [s.strip() for s in skills_data.split(',')]
                top_job_skills.update(skills_list)
            elif isinstance(skills_data, list):
                top_job_skills.update(skills_data)
        
        skill_gaps = top_job_skills - user_skills
        
        if skill_gaps:
            recommendations.append(f"Consider developing these in-demand skills: {', '.join(list(skill_gaps)[:3])}")
        
        return recommendations

    def initialize_job_dataset(self, csv_path: str = "data/job_dataset.csv") -> bool:
        """Initialize job dataset from CSV file - FIXED METADATA TYPES"""
        try:
            if not os.path.exists(csv_path):
                print(f"❌ Job dataset not found at: {csv_path}")
                return False
            
            # Read CSV file
            df = pd.read_csv(csv_path)
            print(f"📊 Loaded job dataset with {len(df)} entries")
            
            # Add jobs directly to vector store
            jobs_added = 0
            
            for index, row in df.iterrows():
                # Create job content
                job_content = f"""
                    Title: {row.get('title', '')}
                    Company: {row.get('company', '')}
                    Location: {row.get('location', '')}
                    Description: {row.get('description', '')}
                    Required Skills: {row.get('required_skills', '')}
                    Experience Level: {row.get('experience_level', '')}
                    Job Type: {row.get('job_type', '')}
                    Salary: {row.get('salary', '')}
                    Posted Date: {row.get('posted_date', '')}
                    """
                
                # Convert skills list to string for metadata
                skills_list = self._parse_skills(row.get('required_skills', ''))
                skills_string = ", ".join(skills_list) if skills_list else ""
                
                # Create metadata - ONLY strings, numbers, or booleans
                job_metadata = {
                    "title": str(row.get('title', '')),
                    "company": str(row.get('company', '')),
                    "location": str(row.get('location', '')),
                    "description": str(row.get('description', ''))[:500],  # Limit length
                    "required_skills": skills_string,  # Convert list to string
                    "experience_level": str(row.get('experience_level', '')),
                    "job_type": str(row.get('job_type', '')),
                    "salary": str(row.get('salary', '')),
                    "min_salary": int(self._extract_min_salary(row.get('salary', ''))),
                    "posted_date": str(row.get('posted_date', '')),
                    "user_id": "system",
                    "doc_type": "job",
                    "source": "job_dataset"
                }
                
                # Add directly to vector store using new method
                chunks_added = self.vector_store.add_document_content(
                    content=job_content,
                    metadata=job_metadata,
                    collection_name="job_descriptions"
                )
                
                jobs_added += 1
                if index % 10 == 0:  # Progress indicator
                    print(f"📝 Added {jobs_added} jobs...")
            
            print(f"✅ Successfully added {jobs_added} jobs to vector database")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize job dataset: {e}")
            return False

    def _parse_skills(self, skills_str: str) -> List[str]:
        """Parse skills string into list"""
        try:
            if pd.isna(skills_str):
                return []
            
            # Handle different skill formats
            skills = []
            for skill in str(skills_str).split(','):
                skill = skill.strip()
                if skill:
                    skills.append(skill)
            
            return skills
        except:
            return []

    def _extract_min_salary(self, salary_str: str) -> int:
        """Extract minimum salary from salary string"""
        try:
            if pd.isna(salary_str):
                return 0
            
            # Handle different salary formats: "$50,000 - $70,000", "50000", etc.
            salary_text = str(salary_str)
            numbers = re.findall(r'\d+', salary_text.replace(',', ''))
            if numbers:
                return int(numbers[0])
        except:
            pass
        
        return 0

    def get_job_by_id(self, job_id: str) -> Dict[str, Any]:
        """Get job by ID from the vector store"""
        try:
            # Search for the specific job
            results = self.vector_store.search("job_descriptions", job_id, n_results=1)
            
            if results:
                # Return the first result with the job_id added
                job_data = results[0]
                job_data["job_id"] = job_id  # Add the ID for reference
                return job_data
            else:
                # Try a broader search
                results = self.vector_store.search("job_descriptions", job_id, n_results=5)
                if results:
                    job_data = results[0]
                    job_data["job_id"] = job_id
                    return job_data
                else:
                    return None
                    
        except Exception as e:
            print(f"❌ Failed to get job by ID {job_id}: {e}")
            return None
        
    def generate_cover_letter(self, job_data: Dict, user_profile: Dict, user_skills: List[str] = None) -> str:
        """Generate a personalized cover letter"""
        try:
            if not job_data:
                return "Error: Job information not available."
            
            # Extract job information from metadata
            metadata = job_data.get("metadata", {})
            job_title = metadata.get("title", "the position")
            company = metadata.get("company", "your company")
            job_description = job_data.get("content", "")
            
            # Get user skills from profile or parameter
            skills = user_skills or user_profile.get("skills", [])
            if isinstance(skills, str):
                skills = [skill.strip() for skill in skills.split(',')]
            
            # Get other user profile information
            experience_level = user_profile.get("experience_level", "Not specified")
            learning_goals = user_profile.get("learning_goals", "Not specified")
            location = user_profile.get("location", "Not specified")
            
            # Create cover letter prompt
            prompt = f"""
                Generate a professional cover letter for a job application with the following details:

                Job Title: {job_title}
                Company: {company}
                Job Description: {job_description[:1000]}

                Applicant Profile:
                - Skills: {', '.join(skills)}
                - Experience Level: {experience_level}
                - Learning Goals: {learning_goals}
                - Location Preference: {location}

                Please write a compelling cover letter that:
                1. Expresses enthusiasm for the specific role and company
                2. Highlights relevant skills from the applicant's profile
                3. Shows how the applicant's background matches the job requirements
                4. Is professional, concise (around 300-400 words), and tailored to this specific opportunity
                5. Includes a call to action for an interview
                6. Uses a professional business letter format

                Cover Letter:
                """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            print(f"❌ Cover letter generation failed: {e}")
            return f"Error generating cover letter: {str(e)}"

class DocumentProcessor:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
    def process_uploaded_documents(self, documents: List[Any], user_id: str = None) -> Dict[str, Any]:
        """Process uploaded documents with error handling and validation"""
        try:
            processed_count = 0
            errors = []
            
            for doc in documents:
                try:
                    # Validate document
                    validation_result = self._validate_document(doc)
                    if not validation_result["valid"]:
                        errors.append(f"Document validation failed: {validation_result['error']}")
                        continue
                    
                    # Extract content based on file type
                    content = self._extract_content(doc)
                    if not content:
                        errors.append(f"No content extracted from document: {doc.get('filename', 'Unknown')}")
                        continue
                    
                    # Create metadata
                    metadata = {
                        "user_id": user_id,
                        "filename": doc.get("filename", "unknown"),
                        "file_type": doc.get("content_type", "unknown"),
                        "processed_at": datetime.now().isoformat(),
                        "doc_type": "knowledge"
                    }
                    
                    # Add directly to vector store using new method
                    chunks_added = self.vector_store.add_document_content(
                        content=content,
                        metadata=metadata,
                        collection_name="knowledge_base"
                    )
                    
                    processed_count += 1
                    print(f"✅ Processed document: {doc.get('filename', 'Unknown')} - {chunks_added} chunks")
                    
                except Exception as e:
                    error_msg = f"Failed to process {doc.get('filename', 'Unknown')}: {str(e)}"
                    errors.append(error_msg)
                    print(f"❌ {error_msg}")
            
            return {
                "processed": processed_count,
                "total": len(documents),
                "errors": errors,
                "success": processed_count > 0
            }
            
        except Exception as e:
            print(f"❌ Document processing failed: {e}")
            return {
                "processed": 0,
                "total": len(documents),
                "errors": [f"Processing system error: {str(e)}"],
                "success": False
            }
    
    def _validate_document(self, document: Any) -> Dict[str, Any]:
        """Validate document before processing"""
        try:
            # Check file size (max 10MB)
            max_size = 10 * 1024 * 1024
            if document.get("size", 0) > max_size:
                return {"valid": False, "error": f"File too large: {document.get('size', 0)} bytes"}
            
            # Check file type
            supported_types = [
                'application/pdf',
                'text/plain',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/msword',
                'text/markdown'
            ]
            
            if document.get("content_type") not in supported_types:
                return {"valid": False, "error": f"Unsupported file type: {document.get('content_type')}"}
            
            return {"valid": True, "error": None}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def _extract_content(self, document: Any) -> str:
        """Extract text content from document based on file type"""
        try:
            content_type = document.get("content_type", "")
            file_content = document.get("content", b"")
            
            if content_type == "application/pdf":
                return self._extract_pdf_content(file_content)
            elif content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                                "application/msword"]:
                return self._extract_docx_content(file_content)
            elif content_type == "text/plain":
                return file_content.decode('utf-8') if isinstance(file_content, bytes) else file_content
            elif content_type == "text/markdown":
                return file_content.decode('utf-8') if isinstance(file_content, bytes) else file_content
            else:
                return ""
                
        except Exception as e:
            print(f"❌ Content extraction failed: {e}")
            return ""
    
    def _extract_pdf_content(self, pdf_content: bytes) -> str:
        """Extract text from PDF files"""
        try:
            import PyPDF2
            from io import BytesIO
            
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            print(f"❌ PDF extraction failed: {e}")
            return ""
    
    def _extract_docx_content(self, docx_content: bytes) -> str:
        """Extract text from DOCX files"""
        try:
            from docx import Document
            from io import BytesIO
            
            doc_file = BytesIO(docx_content)
            doc = Document(doc_file)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
            
        except Exception as e:
            print(f"❌ DOCX extraction failed: {e}")
            return ""
    
    def _chunk_document(self, content: str, filename: str) -> List[Dict]:
        """Split document into manageable chunks"""
        # Simple chunking by paragraphs/sentences
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        chunks = []
        
        current_chunk = ""
        chunk_size = 1000  # characters
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "index": len(chunks)
                })
                current_chunk = ""
            
            current_chunk += paragraph + "\n\n"
        
        # Add the last chunk if any content remains
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "index": len(chunks)
            })
        
        print(f"📄 Chunked {filename} into {len(chunks)} parts")
        return chunks
    
    # def get_job_by_id(self, job_id: str) -> Dict[str, Any]:
    #     """Get job by ID from the vector store"""
    #     try:
    #         # Search for the specific job
    #         results = self.vector_store.search("job_descriptions", job_id, n_results=1)
            
    #         if results:
    #             # Return the first result with the job_id added
    #             job_data = results[0]
    #             job_data["job_id"] = job_id  # Add the ID for reference
    #             return job_data
    #         else:
    #             # Try a broader search
    #             results = self.vector_store.search("job_descriptions", job_id, n_results=5)
    #             if results:
    #                 job_data = results[0]
    #                 job_data["job_id"] = job_id
    #                 return job_data
    #             else:
    #                 return None
                    
    #     except Exception as e:
    #         print(f"❌ Failed to get job by ID {job_id}: {e}")
    #         return None

    # def generate_cover_letter(self, job_data: Dict, user_profile: Dict, user_skills: List[str] = None) -> str:
    #     """Generate a personalized cover letter"""
    #     try:
    #         if not job_data:
    #             return "Error: Job information not available."
            
    #         # Extract job information
    #         job_title = job_data.get("metadata", {}).get("title", "the position")
    #         company = job_data.get("metadata", {}).get("company", "your company")
    #         job_description = job_data.get("content", "")
            
    #         # Get user skills from profile or parameter
    #         skills = user_skills or user_profile.get("skills", [])
    #         if isinstance(skills, str):
    #             skills = [skill.strip() for skill in skills.split(',')]
            
    #         # Create cover letter prompt
    #         prompt = f"""
    #         Generate a professional cover letter for a job application with the following details:

    #         Job Title: {job_title}
    #         Company: {company}
    #         Job Description: {job_description[:1000]}  # Limit length

    #         Applicant Profile:
    #         - Skills: {', '.join(skills)}
    #         - Experience Level: {user_profile.get('experience_level', 'Not specified')}
    #         - Learning Goals: {user_profile.get('learning_goals', 'Not specified')}

    #         Please write a compelling cover letter that:
    #         1. Expresses enthusiasm for the specific role and company
    #         2. Highlights relevant skills from the applicant's profile
    #         3. Shows how the applicant's background matches the job requirements
    #         4. Is professional, concise, and tailored to this specific opportunity
    #         5. Includes a call to action for an interview

    #         Cover Letter:
    #         """
            
    #         response = self.llm.invoke(prompt)
    #         return response.content
            
    #     except Exception as e:
    #         print(f"❌ Cover letter generation failed: {e}")
    #         return f"Error generating cover letter: {str(e)}"