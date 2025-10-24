# 🧠 EduHire.ai  
**A Unified GenAI Career Companion**

EduHire.ai is an **AI-powered personalized career and learning assistant** that integrates *learning recommendation systems*, *job-matching intelligence*, and *automated document generation* to bridge the gap between education and employability.  
It merges intelligent **Learning Agents** and **Job Agents** using a **Retrieval-Augmented Generation (RAG)** framework for adaptive guidance.

---

## 🚀 Key Features

### 🎓 Learning Agent
- Provides **personalized learning recommendations** based on user goals.
- Processes uploaded learning materials (PDF, DOCX, TXT) and builds a contextual vector database.
- Uses RAG pipelines for adaptive learning conversations.

### 💼 Job Agent
- Recommends jobs from a structured dataset (`job_dataset.csv`) and external APIs.
- Performs **semantic matching** based on skills, experience, and learning progress.
- Generates **AI-driven cover letters** tailored to specific job roles.

### ⚙️ Backend
- Built with **Flask** and supports **CORS** for frontend integration.
- Modular architecture with `agents/`, `core/`, and `data/` components.
- Provides multiple endpoints for debugging, job searching, and system initialization.

### 💡 Frontend
- Developed using **Streamlit** for an intuitive interface.
- Integrates seamlessly with backend REST APIs for real-time responses.

---

## 📂 Project Structure

    eduhire.ai/
    ├── data/
    │ ├── job_dataset.csv
    │ └── knowledge_base/
    │ └── Generative_AI_by_Mandava.pdf
    ├── backend/
    │ ├── app.py
    │ ├── agents/
    │ │ ├── job_agent.py
    │ │ └── learning_agent.py
    │ └── core/
    │   ├── rag_engine.py
    │   └── vector_store.py
    └── frontend/
        ├── app.py
        └── .streamlit/
        └── config.toml
    ├── requirements.txt
    └── README.md
    ├── .env
    └── .gitignore


## 🧩 System Architecture

EduHire.ai consists of two major components:
1. **Backend API (Flask)** – Handles RAG, vector embeddings, and intelligent agents.
2. **Frontend UI (Streamlit)** – Enables user interaction with learning and job modules.

### Backend Components
| Module | Description |
|---------|-------------|
| `agents/job_agent.py` | Handles job search, matching, and recommendations. |
| `agents/learning_agent.py` | Provides learning insights and personalized suggestions. |
| `core/rag_engine.py` | Main RAG engine that integrates knowledge retrieval and LLM reasoning. |
| `core/vector_store.py` | Manages vector embeddings for documents and job data. |

---

## 🧠 API Endpoints

### Health & System
| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/` | GET | Welcome message and API info |
| `/health` | GET | System health check |
| `/api/initialize` | POST | Initializes RAG engine and job dataset |

### Knowledge Base & Learning
| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/api/upload-document` | POST | Uploads and processes PDFs or docs |
| `/api/query` | POST | Handles user learning or job queries |
| `/api/learning/recommend` | POST | Returns personalized learning resources |

### Jobs & Applications
| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/api/job/match` | POST | Finds matching jobs based on skills/profile |
| `/api/generate/cover-letter` | POST | Creates personalized AI cover letters |
| `/api/job/api-search` | POST | Fetches jobs from external APIs |
| `/api/job/save` | POST | Saves a job to user’s profile |
| `/api/job/apply` | POST | Applies to a selected job |
| `/api/jobs/all` | GET | Lists all jobs from the dataset |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Ameer1428/EduHire.ai.git
cd eduhire.ai
```


### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Environment Variables
```bash
JOB_API_KEY=your_job_api_key
AZURE_OPENAI_ENDPOINT=your_openai_dpoint
AZURE_OPENAI_MODEL=your_openai_model
AZURE_OPENAI_API_KEY=your_openai_pi_key
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment_name
AZURE_OPENAI_API_VERSION=your_openai_appi_version

```
4️⃣ Run the Backend Server
```bash
cd backend
python app.py
```

### Server will start on:
```aurdino
http://localhost:5000
```

5️⃣ Run the Frontend (Streamlit)cd frontend
```
cd frontend
streamlit run app.py
```

## 🧾 Example API Usage

🧾 Example API Usage
### Initialize System
```bash
POST http://localhost:5000/api/initialize
```

### Get Learning Recommendations
```bash
POST http://localhost:5000/api/learning/recommend
{
  "user_id": "ameer01",
  "learning_goals": ["Generative AI", "Prompt Engineering"]
}
```

### Match Jobs
```bash
POST http://localhost:5000/api/job/match
{
  "user_id": "ameer01",
  "user_profile": {
    "skills": ["Python", "Flask", "LLMs"],
    "experience_level": "Entry"
  }
}
```

## 🧰 Tech Stack
| Layer    | Technology                             |
| -------- | -------------------------------------- |
| Backend  | Python, Flask, dotenv, CORS            |
| Frontend | Streamlit                              |
| ML/RAG   | LangChain, Vector Databases            |
| Data     | CSV-based Job Dataset + Knowledge PDFs |
| Cloud    | Azure (LLM APIs, deployment)     |

## 📈 Future Enhancements

#### 🧩 Integration with LinkedIn & Kaggle APIs for real job postings.

#### 📊 Dashboard for user progress tracking and skill analytics.

#### 💬 Voice and chat-based AI career guidance companion.

#### 🧠 Support for multi-agent collaboration between Learning & Job agents.

## 🧑‍💻 Author

    Ameer Khan
    👩‍💻 GenAI Trainee
    🌐 GitHub: [Ameer1428](https://github.com/Ameer1428)
    • LinkedIn: https://www.linkedin.com/in/ameerkhan1428/

## 🪪 License

This project is licensed under the MIT License – free to use and modify with attribution.