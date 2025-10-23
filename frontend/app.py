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
    page_title="Eduhire.ai - Your AI Career Companion",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #bee5eb;
    }
    .match-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

class EduHireFrontend:
    def __init__(self):
        self.backend_url = BACKEND_URL
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'user_id' not in st.session_state:
            st.session_state.user_id = f"user_{int(time.time())}"
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = {}
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False

    def initialize_system(self):
        """Initialize the backend system"""
        try:
            response = requests.post(f"{self.backend_url}/api/initialize")
            if response.status_code == 200:
                st.session_state.system_initialized = True
                return True
            else:
                st.error("Failed to initialize system")
                return False
        except Exception as e:
            st.error(f"Connection error: {e}")
            return False

    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🚀 Eduhire.ai</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h3 style='color: #666;'>Your Unified GenAI-Powered Career Companion</h3>
            <p>Merge learning personalization with automated job application assistance</p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with user controls"""
        with st.sidebar:
            st.markdown("## 👤 User Profile")
            
            # User ID display
            st.info(f"User ID: `{st.session_state.user_id}`")
            
            # Profile setup
            with st.expander("📝 Setup Your Profile", expanded=True):
                skills = st.text_input("Your Skills (comma-separated)", 
                                     value="Python, Machine Learning, Data Analysis")
                experience = st.selectbox("Experience Level", 
                                        ["Entry", "Intermediate", "Senior", "Expert"])
                learning_goals = st.text_area("Learning Goals", 
                                            "Learn advanced Python, Master ML algorithms, Prepare for data science interviews")
                location = st.text_input("Preferred Location", "Remote")
                
                if st.button("💾 Save Profile"):
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
                if st.button("🔄 Initialize System"):
                    with st.spinner("Initializing Eduhire.ai system..."):
                        if self.initialize_system():
                            st.success("System initialized!")
                            st.rerun()
            
            # Quick actions
            st.markdown("## ⚡ Quick Actions")
            if st.button("🔄 Reset Session"):
                st.session_state.conversation_history = []
                st.rerun()
            
            # Upload documents
            st.markdown("## 📁 Upload Documents")
            uploaded_file = st.file_uploader(
                "Upload resume, certificates, or learning materials",
                type=['pdf', 'docx', 'txt', 'pptx'],
                key="doc_uploader"
            )
            if uploaded_file and st.button("📤 Process Document"):
                self.upload_document(uploaded_file)

    def upload_document(self, file):
        """Upload document to backend"""
        try:
            files = {'document': (file.name, file.getvalue(), file.type)}
            data = {'user_id': st.session_state.user_id, 'type': 'knowledge'}
            
            with st.spinner(f"Processing {file.name}..."):
                response = requests.post(
                    f"{self.backend_url}/api/upload-document",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    st.success(f"✅ {file.name} processed successfully!")
                    st.session_state.uploaded_files.append(file.name)
                else:
                    st.error(f"Failed to process {file.name}")
        except Exception as e:
            st.error(f"Upload error: {e}")

    def render_learning_dashboard(self):
        """Render the learning personalization dashboard - FIXED"""
        try:
            st.markdown('<h2 class="sub-header">🎓 Learning Personalization</h2>', unsafe_allow_html=True)
            
            # Quick status check
            if not st.session_state.get('user_profile'):
                st.warning("⚠️ Please set up your profile in the sidebar first!")
                return
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Learning query interface
                st.markdown("### 💬 Ask Learning Questions")
                learning_query = st.text_area(
                    "Ask about courses, learning paths, or specific topics:",
                    placeholder="e.g., 'What's the best way to learn machine learning?', 'Recommend Python courses for beginners', 'Create a 3-month data science learning plan'",
                    height=100,
                    key="learning_query_input"
                )
                
                if st.button("🎯 Get Learning Recommendations", key="learning_ask"):
                    if learning_query:
                        with st.spinner("🔍 Analyzing your learning needs..."):
                            self.process_learning_query(learning_query)
                    else:
                        st.warning("Please enter a learning question")
                
                # Show conversation history
                if st.session_state.conversation_history:
                    st.markdown("### 📚 Conversation History")
                    for msg in st.session_state.conversation_history[-5:]:  # Show last 5, not reversed
                        with st.chat_message(msg["role"]):
                            st.write(msg["content"])
                            if "sources" in msg and msg["sources"]:
                                with st.expander("View Sources"):
                                    for source in msg["sources"][:3]:
                                        st.write(f"📄 {source.get('content', '')[:200]}...")
            
            with col2:
                # Learning recommendations
                st.markdown("### 📊 Quick Actions")
                
                if st.session_state.user_profile.get('learning_goals'):
                    st.markdown("#### 🎯 Your Goals")
                    goals = st.session_state.user_profile['learning_goals'].split(',')
                    for goal in goals[:3]:
                        st.write(f"• {goal.strip()}")
                
                # Quick learning buttons
                if st.button("🔄 Refresh Recommendations", key="refresh_learn"):
                    self.get_learning_recommendations()
                
                if st.button("📈 Get Learning Plan", key="get_plan"):
                    self.get_learning_recommendations()
                    
        except Exception as e:
            st.error(f"Error loading learning dashboard: {e}")
            st.info("Try setting up your profile first in the sidebar")

    # def render_learning_dashboard(self):
    #     """Render the learning personalization dashboard"""
    #     st.markdown('<h2 class="sub-header">🎓 Learning Personalization</h2>', unsafe_allow_html=True)
        
    #     col1, col2 = st.columns([2, 1])
        
    #     with col1:
    #         # Learning query interface
    #         st.markdown("### 💬 Ask Learning Questions")
    #         learning_query = st.text_area(
    #             "Ask about courses, learning paths, or specific topics:",
    #             placeholder="e.g., 'What's the best way to learn machine learning?', 'Recommend Python courses for beginners', 'Create a 3-month data science learning plan'",
    #             height=100
    #         )
            
    #         if st.button("🎯 Get Learning Recommendations", key="learning_ask"):
    #             if learning_query:
    #                 with st.spinner("🔍 Analyzing your learning needs..."):
    #                     self.process_learning_query(learning_query)
    #             else:
    #                 st.warning("Please enter a learning question")
            
    #         # Show conversation history
    #         if st.session_state.conversation_history:
    #             st.markdown("### 📚 Conversation History")
    #             for msg in reversed(st.session_state.conversation_history[-5:]):  # Show last 5
    #                 with st.chat_message(msg["role"]):
    #                     st.write(msg["content"])
    #                     if "sources" in msg and msg["sources"]:
    #                         with st.expander("View Sources"):
    #                             for source in msg["sources"][:3]:
    #                                 st.write(f"📄 {source.get('content', '')[:200]}...")
        
    #     with col2:
    #         # Learning recommendations
    #         st.markdown("### 📊 Your Learning Dashboard")
            
    #         if st.session_state.user_profile.get('learning_goals'):
    #             st.markdown("#### 🎯 Current Goals")
    #             goals = st.session_state.user_profile['learning_goals'].split(',')
    #             for goal in goals[:3]:
    #                 st.write(f"• {goal.strip()}")
            
    #         # Quick learning actions
    #         st.markdown("#### ⚡ Quick Learning Actions")
    #         if st.button("🔄 Refresh Recommendations"):
    #             self.get_learning_recommendations()
            
    #         if st.button("📈 Progress Report"):
    #             self.show_learning_progress()

    def process_learning_query(self, query: str):
        """Process learning query and display results"""
        try:
            payload = {
                'user_id': st.session_state.user_id,
                'query_type': 'learning',
                'query': query,
                'context': st.session_state.user_profile
            }
            
            response = requests.post(f"{self.backend_url}/api/query", json=payload)
            
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
                    "content": result.get('answer', 'No response'),
                    "sources": result.get('sources', []),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Display learning path if available
                if 'learning_path' in result:
                    self.display_learning_path(result['learning_path'])
                
                st.rerun()
            else:
                st.error("Failed to process query")
                
        except Exception as e:
            st.error(f"Error: {e}")

    def get_learning_recommendations(self):
        """Get personalized learning recommendations"""
        try:
            payload = {
                'user_id': st.session_state.user_id,
                'learning_goals': st.session_state.user_profile.get('learning_goals', '').split(',')
            }
            
            with st.spinner("🎯 Generating personalized recommendations..."):
                response = requests.post(f"{self.backend_url}/api/learning/recommend", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    self.display_recommendations(result)
                else:
                    st.error("Failed to get recommendations")
                    
        except Exception as e:
            st.error(f"Error: {e}")

    def display_recommendations(self, recommendations: Dict):
        """Display learning recommendations"""
        st.markdown("### 🎯 Your Personalized Learning Plan")
        
        rec_data = recommendations.get('recommendations', {})
        
        # Immediate actions
        if 'immediate_actions' in rec_data:
            st.markdown("#### 🚀 Immediate Actions")
            for action in rec_data['immediate_actions']:
                st.write(f"✅ {action}")
        
        # Resource categories
        if 'resource_categories' in rec_data:
            st.markdown("#### 📚 Learning Resources")
            for category, resources in rec_data['resource_categories'].items():
                with st.expander(f"📖 {category.title()} Resources"):
                    for resource in resources:
                        st.write(f"• {resource.get('content', '')[:150]}...")
        
        # Success metrics
        if 'success_metrics' in rec_data:
            st.markdown("#### 🎯 Success Metrics")
            for metric in rec_data['success_metrics']:
                st.write(f"📊 {metric}")

    def display_learning_path(self, learning_path: Dict):
        """Display learning path visualization"""
        st.markdown("### 🗺️ Your Learning Path")
        
        if 'weekly_breakdown' in learning_path:
            # Create a timeline visualization
            weeks = learning_path['weekly_breakdown']
            
            fig = go.Figure()
            
            for i, week in enumerate(weeks):
                fig.add_trace(go.Scatter(
                    x=[i+1] * len(week['topics']),
                    y=week['topics'],
                    mode='markers+text',
                    marker=dict(size=15, color='blue'),
                    text=[f"📚 {t}" for t in week['topics']],
                    textposition="middle right",
                    name=f"Week {i+1}"
                ))
            
            fig.update_layout(
                title="Learning Journey Timeline",
                xaxis_title="Week",
                yaxis_title="Topics",
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def render_job_dashboard(self):
        """Render the job search and application dashboard"""
        st.markdown('<h2 class="sub-header">💼 Job Search & Applications</h2>', unsafe_allow_html=True)
        
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
            
            # Job search filters
            st.text_input("Job Title Keywords", key="job_keywords")
            st.selectbox("Experience Level", 
                        ["Any", "Entry", "Mid", "Senior", "Executive"],
                        key="job_experience")
            st.text_input("Location Preference", 
                         st.session_state.user_profile.get('location', 'Remote'),
                         key="job_location")
            
            if st.button("🚀 Find Matching Jobs", key="find_jobs"):
                self.find_job_matches()
        
        with col2:
            # Job matches will be displayed here
            if 'job_matches' in st.session_state:
                self.display_job_matches(st.session_state.job_matches)

    def find_job_matches(self):
        """Find job matches for user"""
        try:
            payload = {
                'user_id': st.session_state.user_id,
                'user_profile': {
                    **st.session_state.user_profile,
                    'keywords': st.session_state.get('job_keywords', ''),
                    'experience': st.session_state.get('job_experience', ''),
                    'location': st.session_state.get('job_location', '')
                }
            }
            
            with st.spinner("🔍 Finding your perfect job matches..."):
                response = requests.post(f"{self.backend_url}/api/job/match", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.job_matches = result
                    st.rerun()
                else:
                    st.error("Failed to find job matches")
                    
        except Exception as e:
            st.error(f"Error: {e}")

    def display_job_matches(self, matches_data: Dict):
        """Display job matches"""
        matches = matches_data.get('matches', [])
        
        st.markdown(f"### 🎉 Found {len(matches)} Matching Jobs")
        
        for i, job in enumerate(matches[:5]):  # Show top 5 matches
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"#### {job.get('title', 'Unknown Position')}")
                    st.write(f"**Company:** {job.get('company', 'Unknown')}")
                    st.write(f"**Location:** {job.get('location', 'Remote')}")
                    st.write(f"**Match Score:** <span class='match-score'>{job.get('match_score', 0)*100:.1f}%</span>", 
                            unsafe_allow_html=True)
                    
                    # Skills match
                    matching_skills = job.get('matching_skills', [])
                    if matching_skills:
                        st.write("**Matching Skills:**", ", ".join(matching_skills[:3]))
                
                with col2:
                    if st.button(f"📨 Apply", key=f"apply_{i}"):
                        self.generate_cover_letter(job.get('job_id', ''))
                    
                    if st.button(f"💾 Save", key=f"save_{i}"):
                        st.success("Job saved to your list!")
                
                st.divider()

    def render_application_assistant(self):
        """Render application assistant interface"""
        st.markdown("### 🤖 AI Application Assistant")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📄 Cover Letter Generator")
            job_id = st.text_input("Job ID for Cover Letter", placeholder="Enter job ID from your matches")
            job_description = st.text_area("Or paste job description", height=150,
                                         placeholder="Paste the full job description here...")
            
            if st.button("✍️ Generate Cover Letter"):
                if job_id or job_description:
                    self.generate_cover_letter(job_id, job_description)
                else:
                    st.warning("Please provide either Job ID or Job Description")
        
        with col2:
            st.markdown("#### 🎯 Application Tips")
            st.info("""
            **Pro Tips for Better Applications:**
            - Customize each cover letter for the specific job
            - Highlight matching skills from the job description
            - Use keywords from the job posting
            - Keep it concise and professional
            - Proofread before sending
            """)
            
            # Resume improvement suggestions
            if st.button("📊 Analyze Resume Gaps"):
                self.analyze_resume_gaps()

    def generate_cover_letter(self, job_id: str = "", job_description: str = ""):
        """Generate cover letter for a job"""
        try:
            payload = {
                'user_id': st.session_state.user_id,
                'job_id': job_id,
                'job_description': job_description
            }
            
            with st.spinner("✍️ Generating personalized cover letter..."):
                response = requests.post(f"{self.backend_url}/api/generate/cover-letter", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    self.display_cover_letter(result)
                else:
                    st.error("Failed to generate cover letter")
                    
        except Exception as e:
            st.error(f"Error: {e}")

    def display_cover_letter(self, cover_letter_data: Dict):
        """Display generated cover letter"""
        st.markdown("### 📄 Your Personalized Cover Letter")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.text_area("Cover Letter", 
                        cover_letter_data.get('cover_letter', ''),
                        height=400,
                        key="cover_letter_display")
            
            # Action buttons for the cover letter
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                st.download_button(
                    "📥 Download as DOC",
                    cover_letter_data.get('cover_letter', ''),
                    file_name="cover_letter.docx"
                )
            with col1b:
                st.download_button(
                    "📥 Download as PDF",
                    cover_letter_data.get('cover_letter', ''),
                    file_name="cover_letter.txt"
                )
            with col1c:
                if st.button("🔄 Regenerate"):
                    self.generate_cover_letter(
                        cover_letter_data.get('job_id', ''),
                        cover_letter_data.get('job_description', '')
                    )
        
        with col2:
            st.markdown("#### 🎯 Personalization Notes")
            notes = cover_letter_data.get('personalization_notes', [])
            for note in notes:
                st.write(f"✅ {note}")

    def analyze_resume_gaps(self):
        """Analyze resume for skill gaps"""
        st.info("""
        **Resume Analysis Results:**
        - ✅ Strong in Python and Machine Learning
        - 📈 Good project experience
        - 🔍 Consider adding more cloud computing skills
        - 💼 Include more quantifiable achievements
        - 🎯 Add specific metrics to your experience
        """)

    def render_job_analytics(self):
        """Render job search analytics"""
        st.markdown("### 📊 Job Search Analytics")
        
        # Mock analytics data
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Applications Sent", "12", "3 this week")
        
        with col2:
            st.metric("Interview Rate", "25%", "3 interviews")
        
        with col3:
            st.metric("Average Match Score", "78%", "+5% from last month")
        
        # Skills gap analysis
        st.markdown("#### 🔍 Skills Gap Analysis")
        skills_data = {
            'Skill': ['Python', 'Machine Learning', 'Cloud Computing', 'Data Visualization', 'SQL'],
            'Your Level': [85, 78, 45, 65, 70],
            'Market Demand': [90, 85, 75, 70, 80]
        }
        
        df = pd.DataFrame(skills_data)
        fig = px.bar(df, x='Skill', y=['Your Level', 'Market Demand'], 
                    barmode='group', title="Skills vs Market Demand")
        st.plotly_chart(fig, use_container_width=True)

    def show_learning_progress(self):
        """Show learning progress dashboard"""
        st.markdown("### 📈 Learning Progress Dashboard")
        
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
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.line(df, x='Week', y='Skill Improvement',
                          title='Skill Improvement Over Time')
            st.plotly_chart(fig2, use_container_width=True)

    def render_main_dashboard(self):
        """Render the main dashboard"""
        st.markdown("## 🏠 Dashboard Overview")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Learning Goals", "3 Active", "1 Completed")
        
        with col2:
            st.metric("Job Matches", "15", "5 New")
        
        with col3:
            st.metric("Applications", "8 Sent", "2 Interviews")
        
        with col4:
            st.metric("Skills Improved", "75%", "+15%")
        
        # Feature cards
        st.markdown("### 🚀 Quick Access")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.markdown("### 🎓 Learning Hub")
                st.write("Get personalized learning recommendations and track your progress")
                if st.button("Go to Learning", key="learn_btn"):
                    st.session_state.current_page = "learning"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.markdown("### 💼 Job Search")
                st.write("Find matching jobs and get AI-powered application assistance")
                if st.button("Find Jobs", key="jobs_btn"):
                    st.session_state.current_page = "jobs"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            with st.container():
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Analytics")
                st.write("Track your learning progress and job search metrics")
                if st.button("View Analytics", key="analytics_btn"):
                    st.session_state.current_page = "analytics"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
    
    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_sidebar()
        
        # Initialize page state - FIX: Use consistent state management
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "dashboard"
        
        # Navigation - FIX: Use buttons instead of selectbox for better UX
        st.markdown("### 🧭 Navigation")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏠 Dashboard", use_container_width=True):
                st.session_state.current_page = "dashboard"
                st.rerun()
        
        with col2:
            if st.button("🎓 Learning", use_container_width=True):
                st.session_state.current_page = "learning"
                st.rerun()
        
        with col3:
            if st.button("💼 Jobs", use_container_width=True):
                st.session_state.current_page = "jobs"
                st.rerun()
        
        # Render selected page - FIX: Add proper error handling
        try:
            if st.session_state.current_page == "dashboard":
                self.render_main_dashboard()
            elif st.session_state.current_page == "learning":
                self.render_learning_dashboard()
            elif st.session_state.current_page == "jobs":
                self.render_job_dashboard()
        except Exception as e:
            st.error(f"Error loading page: {e}")
            st.info("Please try refreshing the page or go back to dashboard")
            if st.button("🔄 Back to Dashboard"):
                st.session_state.current_page = "dashboard"
                st.rerun()

    # def run(self):
    #     """Main application runner"""
    #     self.render_header()
    #     self.render_sidebar()
        
    #     # Initialize page state
    #     if 'current_page' not in st.session_state:
    #         st.session_state.current_page = "dashboard"
        
    #     # Navigation
    #     page_options = {
    #         "dashboard": "🏠 Dashboard",
    #         "learning": "🎓 Learning",
    #         "jobs": "💼 Jobs",
    #         "analytics": "📊 Analytics"
    #     }
        
    #     # Page selection
    #     selected_page = st.selectbox(
    #         "Navigate to:",
    #         options=list(page_options.keys()),
    #         format_func=lambda x: page_options[x],
    #         key="page_selector"
    #     )
        
    #     st.session_state.current_page = selected_page
        
    #     # Render selected page
    #     if st.session_state.current_page == "dashboard":
    #         self.render_main_dashboard()
    #     elif st.session_state.current_page == "learning":
    #         self.render_learning_dashboard()
    #     elif st.session_state.current_page == "jobs":
    #         self.render_job_dashboard()
    #     elif st.session_state.current_page == "analytics":
    #         self.show_learning_progress()

# Run the application
if __name__ == "__main__":
    app = EduHireFrontend()
    app.run()