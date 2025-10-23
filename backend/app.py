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
CORS(app)

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
    """Generate personalized cover letter - FIXED version"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        job_id = data.get('job_id')
        job_description = data.get('job_description')
        
        if not job_id and not job_description:
            return jsonify({"error": "Either job_id or job_description is required"}), 400
        
        # If job_description is provided, use it directly
        if job_description:
            # Create a mock job data structure
            job_data = {
                "content": job_description,
                "metadata": {
                    "title": "Data Scientist Position",
                    "company": "Target Company", 
                    "description": job_description
                }
            }
            
            # Get user profile (in real app, from database)
            user_profile = {
                "skills": ["Python", "Machine Learning", "GenAI"],
                "experience_level": "Entry",
                "learning_goals": "Learn advanced Python, Master ML algorithms, Prepare for GenAI interviews"
            }
            
            cover_letter = eduhire_system.rag_engine.generate_cover_letter(job_data, user_profile)
        else:
            # Use job_id to find the job
            cover_letter = eduhire_system.job_agent.generate_cover_letter(user_id, job_id)
        
        return jsonify({"cover_letter": cover_letter})
        
    except Exception as e:
        print(f"❌ Cover letter generation error: {e}")
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
    """Debug endpoint to check job data"""
    try:
        # Test search with a simple query
        results = eduhire_system.rag_engine.vector_store.search("job_descriptions", "python", 5)
        
        job_info = []
        for i, job in enumerate(results):
            metadata = job.get("metadata", {})
            job_info.append({
                "index": i,
                "title": metadata.get("title", "No title"),
                "company": metadata.get("company", "No company"),
                "location": metadata.get("location", "No location"),
                "skills": metadata.get("required_skills", "No skills")
            })
        
        return jsonify({
            "total_jobs_in_db": len(results),
            "sample_jobs": job_info,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize system on startup
    print("🚀 Starting EduHire.ai Backend...")
    success = eduhire_system.initialize_system()
    if not success:
        print("❌ System initialization failed! Check the errors above.")
    
    app.run(host='0.0.0.0', port=5000, debug=True)  # ← CHANGE to False