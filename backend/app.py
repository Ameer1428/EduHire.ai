import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Import custom modules
from agents.learning_agent import LearningAgent
from agents.job_agent import JobAgent
from core.vector_store import VectorStore
from core.rag_engine import RAGEngine

load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

class EduHireAI:
    def __init__(self):
        self.vector_store = VectorStore()
        self.rag_engine = RAGEngine(self.vector_store)
        self.learning_agent = LearningAgent(self.rag_engine)
        self.job_agent = JobAgent(self.rag_engine)
        
        # Initialize components
        self.setup_complete = False
    def initialize_system(self):
        """Initialize all system components"""
        try:
            print("🔄 Initializing EduHire.ai system...")
            
            # Initialize vector store with knowledge base
            self.vector_store.initialize_collections()
            
            # Load job dataset using RAG engine
            success = self.rag_engine.initialize_job_dataset("data/job_dataset.csv")
            if not success:
                print("❌ Failed to load job dataset")
                return False
            
            self.setup_complete = True
            print("✅ EduHire.ai system initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False

    def process_user_query(self, user_id, query_type, query, context=None):
        """Route queries to appropriate agent"""
        if not self.setup_complete:
            return {"error": "System not initialized"}
            
        if query_type == "learning":
            return self.learning_agent.process_learning_query(user_id, query, context)
        elif query_type == "job_search":
            return self.job_agent.process_job_query(user_id, query, context)
        else:
            return {"error": "Invalid query type"}

# Global system instance
eduhire_system = EduHireAI()

@app.before_request
def handle_preflight():
    """Handle CORS preflight requests"""
    if request.method == "OPTIONS":
        response = jsonify({"status": "success"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
        return response

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def home():
    """API welcome endpoint"""
    return jsonify({
        "message": "🚀 EduHire.ai Backend API is running!",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/api/query",
            "job_match": "/api/job/match", 
            "learning_recommend": "/api/learning/recommend",
            "cover_letter": "/api/generate/cover-letter",
            "upload": "/api/upload-document"
        },
        "status": "healthy"
    })

@app.route("/health")
def health():
    """Health check endpoint - FIXED"""
    try:
        return jsonify({
            "ok": True,
            "system_initialized": eduhire_system.setup_complete,
            "components": {
                "vector_store": eduhire_system.vector_store is not None,
                "rag_engine": eduhire_system.rag_engine is not None,
                "learning_agent": eduhire_system.learning_agent is not None,
                "job_agent": eduhire_system.job_agent is not None
            },
            "status": "healthy"
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "status": "unhealthy"
        }), 500

@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize the EduHire.ai system"""
    try:
        success = eduhire_system.initialize_system()
        if success:
            return jsonify({"status": "success", "message": "System initialized"})
        else:
            return jsonify({"error": "System initialization failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload-document', methods=['POST'])
def upload_document():
    """Upload knowledge base documents - FIXED field name"""
    try:
        print("📤 Upload request received")
        print(f"📋 Request files: {list(request.files.keys())}")
        print(f"📋 Request form: {list(request.form.keys())}")
        
        # Try different possible file field names
        file = None
        if 'file' in request.files:
            file = request.files['file']
        elif 'document' in request.files:
            file = request.files['document']
        elif 'resume' in request.files:
            file = request.files['resume']
        else:
            # Get the first file if no specific field name
            files = list(request.files.values())
            if files:
                file = files[0]
        
        user_id = request.form.get('user_id', 'default_user')
        
        if not file:
            print("❌ No file found in request")
            return jsonify({"error": "No file provided. Supported field names: 'file', 'document', 'resume'"}), 400
        
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({"error": "No file selected"}), 400
        
        # Read file content
        file_content = file.read()
        filename = file.filename
        
        print(f"📄 Processing file: {filename} ({len(file_content)} bytes) for user: {user_id}")
        
        # Determine content type
        content_type = file.content_type
        if not content_type:
            # Fallback based on extension
            if filename.lower().endswith('.pdf'):
                content_type = 'application/pdf'
            elif filename.lower().endswith('.txt'):
                content_type = 'text/plain'
            elif filename.lower().endswith('.docx'):
                content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            elif filename.lower().endswith('.doc'):
                content_type = 'application/msword'
            elif filename.lower().endswith('.md'):
                content_type = 'text/markdown'
            else:
                content_type = 'application/octet-stream'
        
        print(f"📋 Content type: {content_type}")
        
        # Create document object
        document = {
            "filename": filename,
            "content": file_content,
            "content_type": content_type,
            "size": len(file_content)
        }
        
        # Process document using RAG engine
        result = eduhire_system.rag_engine.process_documents([document], user_id)
        
        print(f"📊 Upload processing result: {result}")
        
        if result.get("success"):
            return jsonify({
                "status": "success",
                "message": f"Document {filename} processed successfully",
                "details": result
            })
        else:
            return jsonify({
                "error": f"Document processing failed: {result.get('errors', ['Unknown error'])}",
                "details": result
            }), 400
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500
    
@app.route('/api/query', methods=['POST'])
def process_query():
    """Main query endpoint for learning and job queries"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        query_type = data.get('query_type')  # 'learning' or 'job_search'
        query = data.get('query')
        context = data.get('context', {})
        
        if not all([user_id, query_type, query]):
            return jsonify({"error": "Missing required fields"}), 400
            
        result = eduhire_system.process_user_query(user_id, query_type, query, context)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Query processing failed: {str(e)}"}), 500

@app.route('/api/job/match', methods=['POST'])
def match_jobs():
    """Find job matches for user - IMPROVED RESPONSE FORMAT"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        user_profile = data.get('user_profile', {})
        
        print(f"🔍 Job match request for user: {user_id}")
        print(f"🎯 User profile: {user_profile}")
        
        # Ensure skills is a list
        if 'skills' in user_profile and isinstance(user_profile['skills'], str):
            user_profile['skills'] = [skill.strip() for skill in user_profile['skills'].split(',')]
        
        # Get matches from job agent
        matches = eduhire_system.job_agent.find_job_matches(user_id, user_profile)
        
        print(f"📊 Job matches found: {matches.get('match_metrics', {}).get('total_found', 0)}")
        
        # Ensure the response has the expected structure
        response_data = {
            "user_id": user_id,
            "matches": matches.get("matches", []),
            "total_found": matches.get("match_metrics", {}).get("total_found", 0),
            "search_query": matches.get("search_query", ""),
            "insights": matches.get("insights", {}),
            "recommendations": matches.get("recommendations", []),
            "status": "success"
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Job match error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "matches": [],
            "total_found": 0,
            "status": "error"
        }), 500

@app.route('/api/learning/recommend', methods=['POST'])
def learning_recommendations():
    """Get personalized learning recommendations"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        user_id = data.get('user_id')
        learning_goals = data.get('learning_goals', [])
        
        # Validate input
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
        
        # Ensure learning_goals is a list
        if isinstance(learning_goals, str):
            learning_goals = [goal.strip() for goal in learning_goals.split(',')]
        
        print(f"🔍 Processing learning recommendations for user: {user_id}")
        print(f"🎯 Goals: {learning_goals}")
        
        recommendations = eduhire_system.learning_agent.get_recommendations(
            user_id, learning_goals
        )
        
        return jsonify({
            "status": "success",
            "recommendations": recommendations
        })
        
    except Exception as e:
        print(f"❌ Error in learning recommendations: {e}")
        return jsonify({
            "error": f"Learning recommendation failed: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/generate/cover-letter', methods=['POST'])
def generate_cover_letter():
    """Generate personalized cover letter - IMPROVED"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        user_id = data.get('user_id')
        job_id = data.get('job_id')
        job_description = data.get('job_description')
        job_data = data.get('job_data')  # Added for frontend compatibility
        
        print(f"📝 Generating cover letter for user: {user_id}")
        
        # If job_data is provided from frontend, use it directly
        if job_data:
            job_title = job_data.get('title', 'the position')
            company = job_data.get('company', 'Target Company')
            job_description = job_data.get('description', job_description)
            
            # Create job data structure for RAG engine
            formatted_job_data = {
                "content": job_description or f"Position: {job_title} at {company}",
                "metadata": {
                    "title": job_title,
                    "company": company,
                    "description": job_description
                }
            }
            
            # Get user profile (in real app, from database)
            user_profile = {
                "skills": data.get('user_skills', ['Python', 'Machine Learning', 'GenAI']),
                "experience_level": data.get('experience_level', 'Entry'),
                "learning_goals": data.get('learning_goals', 'Learn advanced skills')
            }
            
            cover_letter = eduhire_system.rag_engine.generate_cover_letter(
                formatted_job_data, 
                user_profile,
                user_skills=data.get('user_skills')
            )
            
        elif job_description:
            # Create a mock job data structure
            job_data = {
                "content": job_description,
                "metadata": {
                    "title": "Data Scientist Position",
                    "company": "Target Company", 
                    "description": job_description
                }
            }
            
            user_profile = {
                "skills": ["Python", "Machine Learning", "GenAI"],
                "experience_level": "Entry",
                "learning_goals": "Learn advanced Python, Master ML algorithms"
            }
            
            cover_letter = eduhire_system.rag_engine.generate_cover_letter(job_data, user_profile)
        else:
            return jsonify({"error": "Either job_data or job_description is required"}), 400
        
        return jsonify({
            "cover_letter": cover_letter,
            "job_title": job_data.get('metadata', {}).get('title', 'Unknown Position'),
            "company": job_data.get('metadata', {}).get('company', 'Unknown Company')
        })
        
    except Exception as e:
        print(f"❌ Cover letter generation error: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/job/api-search', methods=['POST'])
def api_job_search():
    """Search jobs using external API only"""
    try:
        data = request.get_json()
        query = data.get('query', 'developer')
        location = data.get('location', '')
        
        # Use the job API directly
        api_results = eduhire_system.rag_engine.job_api.search_jobs(query, location)
        
        if api_results.get("success"):
            return jsonify({
                "status": "success",
                "query": query,
                "location": location,
                "jobs": api_results.get("jobs", []),
                "total": api_results.get("total", 0)
            })
        else:
            return jsonify({
                "error": api_results.get("error", "API search failed"),
                "status": "error"
            }), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/debug/learning', methods=['POST'])
def debug_learning():
    """Debug endpoint for learning agent"""
    try:
        data = request.get_json()
        test_query = data.get('query', 'Tell me about machine learning')
        
        # Test the learning agent directly
        result = eduhire_system.learning_agent.process_learning_query(
            "test_user", test_query, {}
        )
        
        return jsonify({
            "status": "success",
            "agent_working": True,
            "result": result
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "agent_working": False,
            "error": str(e)
        }), 500

@app.route('/api/debug/system')
def debug_system():
    """Debug system status"""
    return jsonify({
        "system_initialized": eduhire_system.setup_complete,
        "learning_agent": eduhire_system.learning_agent is not None,
        "job_agent": eduhire_system.job_agent is not None,
        "vector_store": eduhire_system.vector_store is not None,
        "rag_engine": eduhire_system.rag_engine is not None
    })

@app.route('/api/debug/azure-status')
def debug_azure_status():
    """Debug Azure OpenAI connection"""
    try:
        # Test the RAG engine's LLM
        test_response = eduhire_system.rag_engine.llm.invoke("Say 'Hello World'")
        
        return jsonify({
            "status": "success",
            "azure_connected": True,
            "test_response": str(test_response.content)[:200],
            "model": "Working correctly"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "azure_connected": False,
            "error": str(e)
        }), 500
    
@app.route('/api/debug/jobs', methods=['GET'])
def debug_jobs():
    """Debug endpoint to check job data in database"""
    try:
        # Test search with different queries
        test_queries = ["python", "machine learning", "data", "developer", "ai"]
        
        debug_info = {
            "collections": list(eduhire_system.vector_store.collections.keys()),
            "test_searches": {}
        }
        
        for query in test_queries:
            try:
                results = eduhire_system.vector_store.search("job_descriptions", query, 5)
                debug_info["test_searches"][query] = {
                    "found": len(results),
                    "sample_titles": [job.get("metadata", {}).get("title", "No title") for job in results[:3]]
                }
            except Exception as e:
                debug_info["test_searches"][query] = {"error": str(e)}
        
        # Also check what's actually in the collection
        try:
            # Try to get all jobs (with a high limit)
            all_jobs = eduhire_system.vector_store.search("job_descriptions", "", 100)
            debug_info["all_jobs_sample"] = [
                {
                    "title": job.get("metadata", {}).get("title", "No title"),
                    "company": job.get("metadata", {}).get("company", "No company"),
                    "location": job.get("metadata", {}).get("location", "No location"),
                    "skills": job.get("metadata", {}).get("required_skills", "No skills")
                }
                for job in all_jobs[:5]  # Show first 5 jobs
            ]
            debug_info["total_jobs_in_db"] = len(all_jobs)
        except Exception as e:
            debug_info["all_jobs_error"] = str(e)
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/test/job-search', methods=['POST'])
def test_job_search():
    """Test job search with simple query"""
    try:
        data = request.get_json()
        query = data.get('query', 'python')
        
        print(f"🧪 Testing job search with query: '{query}'")
        
        # Direct vector store search
        results = eduhire_system.vector_store.search("job_descriptions", query, 10)
        
        response = {
            "test_query": query,
            "results_found": len(results),
            "jobs": []
        }
        
        for i, job in enumerate(results):
            metadata = job.get("metadata", {})
            response["jobs"].append({
                "rank": i + 1,
                "title": metadata.get("title", "No title"),
                "company": metadata.get("company", "No company"),
                "content_preview": job.get("content", "")[:100] + "..."
            })
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/csv-structure', methods=['GET'])
def debug_csv_structure():
    """Check the actual CSV file structure"""
    try:
        csv_path = "data/job_dataset.csv"
        if not os.path.exists(csv_path):
            return jsonify({"error": f"CSV file not found at {csv_path}"}), 400
        
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        return jsonify({
            "csv_exists": True,
            "total_rows": len(df),
            "columns": list(df.columns),
            "sample_data": df.head(3).to_dict('records'),
            "dtypes": dict(df.dtypes)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/csv-sample', methods=['GET'])
def debug_csv_sample():
    """Check actual CSV data sample"""
    try:
        csv_path = "data/job_dataset.csv"
        if not os.path.exists(csv_path):
            return jsonify({"error": f"CSV file not found at {csv_path}"}), 400
        
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        # Show first 3 rows with all columns
        sample_data = []
        for i in range(min(3, len(df))):
            row_data = {}
            for col in df.columns:
                value = df.iloc[i][col]
                # Convert numpy types to Python native types for JSON serialization
                if pd.isna(value):
                    row_data[col] = None
                elif hasattr(value, 'item'):  # For numpy types
                    row_data[col] = value.item() if hasattr(value, 'item') else str(value)
                else:
                    row_data[col] = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            sample_data.append(row_data)
        
        return jsonify({
            "csv_exists": True,
            "total_rows": len(df),
            "columns": list(df.columns),
            "sample_data": sample_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/api-raw', methods=['POST'])
def debug_api_raw():
    """Debug raw API results without filtering"""
    try:
        data = request.get_json()
        query = data.get('query', 'Python developer')
        location = data.get('location', '')
        
        # Get raw API results
        api_results = eduhire_system.rag_engine.job_api.search_jobs(query, location)
        
        if api_results.get("success"):
            # Show raw jobs before any filtering
            raw_jobs = []
            for job in api_results.get("jobs", []):
                metadata = job.get("metadata", {})
                raw_jobs.append({
                    "title": metadata.get("title"),
                    "company": metadata.get("company"),
                    "location": metadata.get("location"),
                    "experience_level": metadata.get("experience_level"),
                    "skills": metadata.get("required_skills"),
                    "source": job.get("source")
                })
            
            return jsonify({
                "status": "success",
                "raw_api_results": raw_jobs,
                "total_raw": len(raw_jobs),
                "query_used": query,
                "location_used": location
            })
        else:
            return jsonify({
                "error": api_results.get("error", "API failed"),
                "status": "error"
            }), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/debug/all-jobs', methods=['POST'])
def debug_all_jobs():
    """Debug endpoint to see ALL available jobs"""
    try:
        data = request.get_json()
        user_profile = data.get('user_profile', {})
        
        # Get all jobs from both sources
        all_local_jobs = eduhire_system.vector_store.search("job_descriptions", "", 100)
        api_results = eduhire_system.rag_engine.job_api.search_jobs("Python", "hyderabad")
        api_jobs = api_results.get("jobs", []) if api_results.get("success") else []
        
        response_data = {
            "local_jobs_count": len(all_local_jobs),
            "api_jobs_count": len(api_jobs),
            "local_jobs_sample": [
                {
                    "title": job.get("metadata", {}).get("title"),
                    "company": job.get("metadata", {}).get("company"),
                    "location": job.get("metadata", {}).get("location")
                }
                for job in all_local_jobs[:5]
            ],
            "api_jobs_sample": [
                {
                    "title": job.get("metadata", {}).get("title"),
                    "company": job.get("metadata", {}).get("company"), 
                    "location": job.get("metadata", {}).get("location")
                }
                for job in api_jobs[:5]
            ]
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/jobs/all', methods=['GET'])
def get_all_jobs():
    """Get all jobs directly from database - FOR DEBUGGING"""
    try:
        # Get all jobs without search
        all_jobs = eduhire_system.vector_store.search("job_descriptions", "", 100)
        
        formatted_jobs = []
        for job in all_jobs:
            metadata = job.get("metadata", {})
            formatted_jobs.append({
                "title": metadata.get("title", "No Title"),
                "company": metadata.get("company", "No Company"),
                "location": metadata.get("location", "No Location"),
                "skills": metadata.get("required_skills", "No Skills"),
                "experience_level": metadata.get("experience_level", "Not specified"),
                "description_preview": job.get("content", "")[:100] + "..."
            })
        
        return jsonify({
            "total_jobs": len(formatted_jobs),
            "jobs": formatted_jobs
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/job/save', methods=['POST'])
def save_job():
    """Save a job for later"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        job_id = data.get('job_id')
        job_data = data.get('job_data')
        
        if not all([user_id, job_id, job_data]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # In a real app, you'd save to a database
        # For now, we'll just log and return success
        print(f"💾 User {user_id} saved job {job_id}: {job_data.get('title', 'Unknown')}")
        
        return jsonify({
            "status": "success",
            "message": "Job saved successfully",
            "job_id": job_id
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/job/apply', methods=['POST'])
def apply_to_job():
    """Apply to a job"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        job_id = data.get('job_id')
        job_data = data.get('job_data')
        cover_letter = data.get('cover_letter', '')
        
        if not all([user_id, job_id, job_data]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # In a real app, you'd save application to database and maybe send email
        print(f"📨 User {user_id} applied to job {job_id}: {job_data.get('title', 'Unknown')}")
        print(f"📝 Cover letter length: {len(cover_letter)} characters")
        
        return jsonify({
            "status": "success", 
            "message": "Application submitted successfully",
            "job_id": job_id,
            "application_id": f"app_{int(datetime.now().timestamp())}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/saved-jobs', methods=['GET'])
def get_saved_jobs():
    """Get user's saved jobs"""
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
        
        # In a real app, you'd fetch from database
        # For now, return empty list
        return jsonify({
            "saved_jobs": [],
            "total": 0
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/job/details', methods=['GET'])
def get_job_details():
    """Get detailed job information"""
    try:
        job_id = request.args.get('job_id')
        if not job_id:
            return jsonify({"error": "job_id is required"}), 400
        
        # For API jobs, we need to handle them differently
        if job_id.startswith('api_'):
            # This would require storing API job details
            return jsonify({
                "error": "API job details not stored locally",
                "job_id": job_id
            })
        else:
            # For local jobs, search in vector store
            results = eduhire_system.vector_store.search("job_descriptions", job_id, 1)
            if results:
                return jsonify({
                    "job": results[0],
                    "status": "success"
                })
            else:
                return jsonify({"error": "Job not found"}), 404
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize system on startup
    print("🚀 Starting EduHire.ai Backend...")
    success = eduhire_system.initialize_system()
    if not success:
        print("❌ System initialization failed! Check the errors above.")
    
    app.run(host='0.0.0.0', port=5000, debug=True)  # ← CHANGE to False