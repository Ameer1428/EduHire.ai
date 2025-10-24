import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
from typing import Dict, List, Any

# Configuration
BACKEND_URL = "http://localhost:5000"
st.set_page_config(
    page_title="Eduhire.ai - AI Career Platform",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Netflix-style Modern UI ---
st.markdown("""
<style>
    :root {
        --primary-color: transparent !important;
        --background-color: #0f0f0f !important;
        --text-color: #f1f1f1 !important;
    }
    body, .stApp, [data-testid="stAppViewContainer"] {
        background-color: #0f0f0f !important;
        color: #f1f1f1 !important;
    }
    /* General App Layout */
    .stApp {
        background-color: #0f0f0f;
        color: #f1f1f1;
        font-family: 'Poppins', sans-serif;
    }

    /* Font imports */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;600;700&display=swap');

    /* Headings */
    .main-header {
        font-size: 4rem;
        text-align: center;
        color: #ff2b2b;
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(255, 0, 0, 0.6);
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #ccc;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.7rem;
        color: #fff;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        margin: 2rem 0 1rem 0;
        border-left: 5px solid #ff2b2b;
        padding-left: 15px;
        text-shadow: 0 0 8px rgba(255, 43, 43, 0.4);
    }

    /* Cards */
    .netflix-card {
        background: linear-gradient(135deg, #1b1b1b, #2c2c2c);
        border-radius: 14px;
        padding: 1.5rem;
        color: #f1f1f1;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .netflix-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(255, 43, 43, 0.25);
    }

    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, #202020, #161616);
        padding: 1.5rem;
        border-radius: 14px;
        border-left: 4px solid #ff2b2b;
        color: #f1f1f1;
        box-shadow: 0 2px 15px rgba(255, 43, 43, 0.15);
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        border-left-color: #ff5a5a;
        transform: translateY(-2px);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #ff2b2b, #b81d24);
        color: #fff;
        border: none;
        border-radius: 6px;
        font-size: 1rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        padding: 0.7rem 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 0 8px rgba(255, 43, 43, 0.4);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #ff4f4f, #ff2b2b);
        transform: translateY(-3px);
        box-shadow: 0 0 15px rgba(255, 43, 43, 0.6);
    }

    /* Inputs & Selects */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select {
        background-color: #1b1b1b !important;
        color: #f1f1f1 !important;
        border: 1px solid #333 !important;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    .stSidebar {
        background-color: #121212 !important;
        color: #e0e0e0 !important;
    }

    /* Metrics & Cards inside columns */
    div[data-testid="stMetricValue"] {
        color: #ff2b2b !important;
    }

    /* Remove Streamlit default clutter */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Fix for dark text on dark background */
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #f1f1f1 !important;
    }
    .stMarkdown {
        color: #f1f1f1 !important;
}

</style>
""", unsafe_allow_html=True)


class EduHireFrontend:
    def __init__(self):
        self.backend_url = BACKEND_URL
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        default_states = {
            'user_id': f"user_{int(time.time())}",
            'user_profile': {},
            'conversation_history': [],
            'uploaded_files': [],
            'system_initialized': False,
            'job_matches': [],
            'current_page': "dashboard"
        }
        
        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def initialize_system(self):
        """Initialize the backend system with better feedback"""
        try:
            with st.spinner("🚀 Starting EduHire.ai System..."):
                response = requests.post(f"{self.backend_url}/api/initialize", timeout=30)
                
                if response.status_code == 200:
                    st.session_state.system_initialized = True
                    
                    # Test backend connection
                    try:
                        health_response = requests.get(f"{self.backend_url}/health", timeout=10)
                        if health_response.status_code == 200:
                            return True
                        else:
                            st.warning("⚠️ System initialized but backend connection unstable")
                            return True
                    except:
                        st.warning("⚠️ System initialized but backend health check failed")
                        return True
                else:
                    st.error("❌ Failed to initialize system backend")
                    return False
        except Exception as e:
            st.error(f"🔌 Connection error: {e}")
            st.info("💡 Make sure the backend server is running: `python app.py` in your backend folder")
            return False

    def render_header(self):
        """Render the main app title and subtitle"""
        st.markdown("""
        <div style="
            text-align: center;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        ">
            <h1 style="
                font-size: 3.5rem;
                color: #ff2b2b;
                font-weight: 800;
                letter-spacing: 2px;
                text-shadow: 0 0 15px rgba(255, 0, 0, 0.6);
                font-family: 'Poppins', sans-serif;
            ">
                EduHire.ai
            </h1>
        </div>

        <div style="
            background: linear-gradient(90deg, #ff2b2b 0%, #ff5555 100%);
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            margin: 0 auto 2rem auto;
            width: 85%;
            text-align: center;
            font-weight: 600;
            font-size: 1.1rem;
            box-shadow: 0 0 20px rgba(255, 43, 43, 0.4);
        ">
            🚀 Empower your career journey with <strong>EduHire.ai</strong> — Learn, Grow, and Get Hired with GenAI!
        </div>
        """, unsafe_allow_html=True)

    def render_navigation(self):
        """Render Netflix-style navigation"""
        cols = st.columns(4)
        pages = [
            ("🏠", "Dashboard", "dashboard"),
            ("🎓", "Learning", "learning"), 
            ("💼", "Jobs", "jobs"),
            ("📊", "Analytics", "analytics")
        ]
        
        for i, (icon, label, page) in enumerate(pages):
            with cols[i]:
                is_active = st.session_state.current_page == page
                button_label = f"**{icon} {label}**" if is_active else f"{icon} {label}"
                if st.button(button_label, use_container_width=True, key=f"nav_{page}"):
                    st.session_state.current_page = page
                    st.rerun()

    def render_sidebar(self):
        """Render the sidebar with user controls"""
        with st.sidebar:
            st.markdown("## 👤 User Profile")
            st.info(f"User ID: `{st.session_state.user_id}`")
            
            # Profile setup
            with st.expander("📝 Setup Your Profile", expanded=True):
                skills = st.text_input("Your Skills (comma-separated)", 
                                     placeholder="e.g., GenAI, Python, Cloud Computing, Data Analysis")
                experience = st.selectbox("Experience Level", 
                                        ["Select Level", "Entry", "Intermediate", "Senior", "Expert"])
                learning_goals = st.text_area("Learning Goals", 
                                            placeholder="e.g., Learn GenAI technologies, Master cloud platforms, Prepare for AI engineering roles")
                location = st.text_input("Preferred Location", "Remote")
                
                if st.button("💾 Save Profile", use_container_width=True):
                    if not skills or experience == "Select Level":
                        st.error("Please fill in Skills and select Experience Level")
                    else:
                        st.session_state.user_profile = {
                            "skills": [s.strip() for s in skills.split(",")],
                            "experience_level": experience.lower(),
                            "learning_goals": learning_goals,
                            "location": location
                        }
                        st.success("Profile saved successfully!")
            
            # System status
            st.markdown("## 🔧 System Status")
            if st.session_state.system_initialized:
                st.success("✅ System Ready")
            else:
                st.warning("⚠️ System Not Initialized")
                if st.button("🔄 Initialize System", use_container_width=True):
                    if self.initialize_system():
                        st.rerun()
            
            # Quick actions
            st.markdown("## ⚡ Quick Actions")
            if st.button("🔄 Reset Session", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key not in ['user_id']:
                        del st.session_state[key]
                st.rerun()
            
            # Upload documents
            st.markdown("## 📁 Upload Documents")
            uploaded_file = st.file_uploader(
                "Upload resume, certificates, or learning materials",
                type=['pdf', 'docx', 'txt', 'pptx'],
                key="doc_uploader"
            )
            if uploaded_file and st.button("📤 Process Document", use_container_width=True):
                self.upload_document(uploaded_file)

    def upload_document(self, file):
        """Upload document to backend with better feedback"""
        try:
            files = {'document': (file.name, file.getvalue(), file.type)}
            data = {'user_id': st.session_state.user_id, 'type': 'resume'}
            
            with st.spinner(f"📤 Processing {file.name}..."):
                response = requests.post(
                    f"{self.backend_url}/api/upload-document",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ {file.name} processed successfully!")
                    st.session_state.uploaded_files.append({
                        'name': file.name,
                        'type': file.type,
                        'size': len(file.getvalue()),
                        'processed_at': datetime.now().isoformat()
                    })
                else:
                    st.error(f"❌ Failed to process {file.name}. Status: {response.status_code}")
                    
        except Exception as e:
            st.error(f"📤 Upload error: {e}")

    def render_dashboard(self):
        """Render the main dashboard"""
        st.markdown('<div class="section-header">Career Overview</div>', unsafe_allow_html=True)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="netflix-card" style="text-align: center;">
                <div style="font-size: 2rem;">🎯</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #ffffff;">
                    """ + ("3 Active" if st.session_state.user_profile else "0 Active") + """
                </div>
                <div style="color: #b3b3b3;">Learning Goals</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="netflix-card" style="text-align: center;">
                <div style="font-size: 2rem;">💼</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #ffffff;">
                    """ + (str(len(st.session_state.job_matches)) if st.session_state.job_matches else "0") + """
                </div>
                <div style="color: #b3b3b3;">Job Matches</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="netflix-card" style="text-align: center;">
                <div style="font-size: 2rem;">📨</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #ffffff;">0</div>
                <div style="color: #b3b3b3;">Applications</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="netflix-card" style="text-align: center;">
                <div style="font-size: 2rem;">⚡</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #ffffff;">
                    """ + ("75%" if st.session_state.user_profile else "0%") + """
                </div>
                <div style="color: #b3b3b3;">Profile Complete</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature cards
        st.markdown('<div class="section-header">Quick Access</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🎓 Learning Hub")
            st.write("Get personalized learning recommendations and track your progress")
            if st.button("Go to Learning", key="learn_btn", use_container_width=True):
                st.session_state.current_page = "learning"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 💼 Job Search")
            st.write("Find matching jobs and get AI-powered application assistance")
            if st.button("Find Jobs", key="jobs_btn", use_container_width=True):
                st.session_state.current_page = "jobs"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Analytics")
            st.write("Track your learning progress and job search metrics")
            if st.button("View Analytics", key="analytics_btn", use_container_width=True):
                st.session_state.current_page = "analytics"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    def render_learning_dashboard(self):
        """Render the learning personalization dashboard"""
        st.markdown('<div class="section-header">🎓 Learning Personalization</div>', unsafe_allow_html=True)
        
        if not st.session_state.get('user_profile'):
            st.warning("⚠️ Please set up your profile in the sidebar first!")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 💬 Ask Learning Questions")
            learning_query = st.text_area(
                "Ask about courses, learning paths, or specific topics:",
                placeholder="e.g., 'What GenAI courses do you recommend?', 'Create a learning plan for cloud computing', 'How to become an AI engineer'",
                height=100,
                key="learning_query_input"
            )
            
            if st.button("🎯 Get Learning Recommendations", key="learning_ask", use_container_width=True):
                if learning_query:
                    with st.spinner("🔍 Analyzing your learning needs..."):
                        self.process_learning_query(learning_query)
                else:
                    st.warning("Please enter a learning question")
            
            # Show conversation history
            if st.session_state.conversation_history:
                st.markdown("### 📚 Conversation History")
                for msg in st.session_state.conversation_history[-5:]:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
        
        with col2:
            st.markdown("### 📊 Quick Actions")
            
            if st.session_state.user_profile.get('learning_goals'):
                st.markdown("#### 🎯 Your Goals")
                goals = st.session_state.user_profile['learning_goals'].split(',')
                for goal in goals[:3]:
                    st.markdown(f"<div style='color: #b3b3b3; margin: 0.5rem 0;'>• {goal.strip()}</div>", unsafe_allow_html=True)
            
            if st.button("🔄 Refresh Recommendations", key="refresh_learn", use_container_width=True):
                self.get_learning_recommendations()

    def process_learning_query(self, query: str):
        """Process learning query and display results"""
        try:
            payload = {
                'user_id': st.session_state.user_id,
                'query_type': 'learning',
                'query': query,
                'context': st.session_state.user_profile
            }
            
            response = requests.post(f"{self.backend_url}/api/query", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    "role": "user", 
                    "content": query,
                    "timestamp": datetime.now().isoformat()
                })
                
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": result.get('answer', 'No response received from AI.'),
                    "timestamp": datetime.now().isoformat()
                })
                
                st.rerun()
            else:
                st.error("Failed to process query. Backend may not be ready.")
                
        except Exception as e:
            st.error(f"Error processing query: {e}")

    def get_learning_recommendations(self):
        """Get personalized learning recommendations"""
        try:
            if not st.session_state.user_profile.get('learning_goals'):
                st.error("Please set your learning goals in your profile first")
                return
                
            payload = {
                'user_id': st.session_state.user_id,
                'learning_goals': st.session_state.user_profile.get('learning_goals', '').split(',')
            }
            
            with st.spinner("🎯 Generating personalized recommendations..."):
                response = requests.post(f"{self.backend_url}/api/learning/recommend", json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Recommendations generated!")
                    # You can display the result here
                else:
                    st.error("Backend not responding. Please ensure system is initialized.")
                    
        except Exception as e:
            st.error(f"Error: {e}")

    def render_job_dashboard(self):
        """Render the job search and application dashboard"""
        st.markdown('<div class="section-header">💼 Job Search & Applications</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🔍 Job Matching", "📝 Application Assistant", "📊 Job Analytics"])
        
        with tab1:
            self.render_job_matching()
        with tab2:
            self.render_application_assistant()
        with tab3:
            self.render_job_analytics()

    def render_job_matching(self):
        """Render job matching interface"""
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎯 Find Your Dream Job")
            
            job_keywords = st.text_input("Job Title Keywords", 
                                       value=st.session_state.user_profile.get('skills', ['GenAI'])[0] if st.session_state.user_profile else "GenAI")
            job_experience = st.selectbox("Experience Level", 
                        ["Any", "Entry", "Mid", "Senior", "Executive"],
                        key="job_experience")
            job_location = st.text_input("Location Preference", 
                         st.session_state.user_profile.get('location', 'Remote'),
                         key="job_location")
            
            if st.button("🚀 Find Matching Jobs", key="find_jobs", use_container_width=True):
                self.find_job_matches(job_keywords, job_experience, job_location)
        
        with col2:
            if st.session_state.job_matches:
                self.display_job_matches(st.session_state.job_matches)
            else:
                st.markdown("""
                <div class="netflix-card" style="text-align: center; padding: 3rem;">
                    <div style="font-size: 4rem;">🔍</div>
                    <h3>Find Your Perfect Job Match</h3>
                    <p style="color: #b3b3b3;">Use the search filters to find jobs that match your profile</p>
                </div>
                """, unsafe_allow_html=True)

    def find_job_matches(self, keywords="", experience="", location=""):
        """Find job matches for user"""
        try:
            payload = {
                'user_id': st.session_state.user_id,
                'user_profile': {
                    **st.session_state.user_profile,
                    'keywords': keywords,
                    'experience': experience,
                    'location': location
                }
            }
            
            with st.spinner("🔍 Finding your perfect job matches..."):
                response = requests.post(f"{self.backend_url}/api/job/match", json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.job_matches = result.get('matches', [])
                    if st.session_state.job_matches:
                        st.success(f"🎉 Found {len(st.session_state.job_matches)} matching jobs!")
                    else:
                        st.info("🤔 No matches found. Try adjusting your search criteria.")
                    st.rerun()
                else:
                    st.error("Backend not responding. Please ensure system is initialized.")
                    
        except Exception as e:
            st.error(f"Error searching jobs: {e}")

    def display_job_matches(self, matches: List):
        """Display job matches"""
        st.markdown(f"### 🎉 Found {len(matches)} Matching Jobs")
        
        for i, job in enumerate(matches[:5]):
            with st.container():
                st.markdown(f"""
                <div class="netflix-card">
                    <h3 style="color: #ffffff; margin-bottom: 0.5rem;">{job.get('title', 'Unknown Position')}</h3>
                    <p style="color: #b3b3b3; margin: 0.25rem 0;">
                        <strong>Company:</strong> {job.get('company', 'Unknown')} • 
                        <strong>Location:</strong> {job.get('location', 'Remote')}
                    </p>
                    <p style="color: #28a745; font-weight: bold; margin: 0.5rem 0;">
                        Match Score: {job.get('match_score', 0)*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"📨 Apply", key=f"apply_{i}", use_container_width=True):
                        st.info("Apply functionality would open here")
                with col2:
                    if st.button(f"💾 Save", key=f"save_{i}", use_container_width=True):
                        st.success("Job saved to your list!")
                
                if i < len(matches[:5]) - 1:
                    st.markdown("---")

    def render_application_assistant(self):
        """Render application assistant interface"""
        st.markdown("### 🤖 AI Application Assistant")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📄 Cover Letter Generator")
            job_description = st.text_area("Paste job description", height=150,
                                         placeholder="Paste the full job description here to generate a customized cover letter...")
            
            if st.button("✍️ Generate Cover Letter", use_container_width=True):
                if job_description:
                    self.generate_cover_letter(job_description)
                else:
                    st.warning("Please provide a Job Description")
        
        with col2:
            st.markdown("#### 🎯 Application Tips")
            st.markdown("""
            <div class="netflix-card">
            **Pro Tips for Better Applications:**
            - Customize each cover letter for the specific job
            - Highlight matching skills from the job description  
            - Use keywords from the job posting
            - Keep it concise and professional
            - Proofread before sending
            </div>
            """, unsafe_allow_html=True)

    def generate_cover_letter(self, job_description: str = ""):
        """Generate cover letter for a job"""
        try:
            payload = {
                'user_id': st.session_state.user_id,
                'job_description': job_description,
                'user_profile': st.session_state.user_profile
            }
            
            with st.spinner("✍️ Generating personalized cover letter..."):
                response = requests.post(f"{self.backend_url}/api/generate/cover-letter", json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    self.display_cover_letter(result)
                else:
                    st.error("Failed to generate cover letter. Backend may not be ready.")
                    
        except Exception as e:
            st.error(f"Error: {e}")

    def display_cover_letter(self, cover_letter_data: Dict):
        """Display generated cover letter"""
        st.markdown("### 📄 Your Personalized Cover Letter")
        
        cover_letter_text = cover_letter_data.get('cover_letter', 'No cover letter generated. This is a demo response. In a full implementation, AI would generate a customized cover letter based on your profile and the job description.')
        
        st.text_area("Cover Letter", 
                    cover_letter_text,
                    height=400,
                    key="cover_letter_display")
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download as DOC",
                cover_letter_text,
                file_name="cover_letter.docx",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "📥 Download as TXT", 
                cover_letter_text,
                file_name="cover_letter.txt",
                use_container_width=True
            )

    def render_job_analytics(self):
        """Render job search analytics"""
        st.markdown("### 📊 Job Search Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Applications Sent", "0", "0 this week")
        with col2:
            st.metric("Interview Rate", "0%", "0 interviews")
        with col3:
            matches_count = len(st.session_state.job_matches)
            st.metric("Job Matches", matches_count, f"{matches_count} found")
        
        # Skills gap analysis
        st.markdown("#### 🔍 Skills Gap Analysis")
        if st.session_state.user_profile.get('skills'):
            user_skills = st.session_state.user_profile['skills']
            skills_data = {
                'Skill': user_skills + ['Cloud Computing', 'GenAI'],
                'Your Level': [85, 78, 65, 45, 90][:len(user_skills)+2],
                'Market Demand': [90, 85, 80, 75, 95][:len(user_skills)+2]
            }
            
            df = pd.DataFrame(skills_data)
            fig = px.bar(df, x='Skill', y=['Your Level', 'Market Demand'], 
                        barmode='group', title="Your Skills vs Market Demand")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)', 
                font_color='#e5e5e5',
                title_font_color='#ffffff'
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_analytics_page(self):
        """Show learning progress dashboard"""
        st.markdown('<div class="section-header">📈 Learning Progress Dashboard</div>', unsafe_allow_html=True)
        
        if not st.session_state.user_profile:
            st.warning("Please set up your profile to see analytics")
            return
            
        # Mock progress data
        progress_data = {
            'Week': [1, 2, 3, 4, 5, 6],
            'Topics Completed': [2, 4, 7, 10, 12, 15],
            'Hours Studied': [8, 15, 22, 30, 35, 42],
            'Skill Improvement': [10, 25, 40, 55, 65, 75]
        }
        
        df = pd.DataFrame(progress_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.line(df, x='Week', y='Topics Completed', 
                          title='Learning Progress Over Time')
            fig1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e5e5e5',
                title_font_color='#ffffff'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.line(df, x='Week', y='Skill Improvement',
                          title='Skill Improvement Over Time')
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e5e5e5', 
                title_font_color='#ffffff'
            )
            st.plotly_chart(fig2, use_container_width=True)

    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_sidebar()
        self.render_navigation()
        
        # Render selected page
        try:
            if st.session_state.current_page == "dashboard":
                self.render_dashboard()
            elif st.session_state.current_page == "learning":
                self.render_learning_dashboard()
            elif st.session_state.current_page == "jobs":
                self.render_job_dashboard()
            elif st.session_state.current_page == "analytics":
                self.render_analytics_page()
        except Exception as e:
            st.error(f"Error loading page: {e}")
            if st.button("🔄 Back to Dashboard"):
                st.session_state.current_page = "dashboard"
                st.rerun()

# Run the application
if __name__ == "__main__":
    app = EduHireFrontend()
    app.run()