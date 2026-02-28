
# 🏥 MediBot - AI Medical Chatbot

An intelligent AI-powered Medical Chatbot built using RAG (Retrieval Augmented Generation) architecture. The chatbot answers medical questions by searching through a medical encyclopedia PDF containing 759 pages. It uses LangChain for the AI pipeline, Groq LLM (Llama 3.3 70B) for generating answers, FAISS vector database for storing and searching medical knowledge, FastAPI for the backend API, and SQLite database for storing all conversations automatically.

## 🚀 Live Demo
- **API URL:** http://localhost:8000
- **Swagger UI:** http://localhost:8000/docs
- **GitHub:** https://github.com/Panchasheelag89/medical-chatbot

## ✨ Features
- 🔍 Ask any medical question in plain English
- 📚 Answers from a 759-page medical encyclopedia
- 🤖 Powered by Groq LLM (Llama 3.3 70B)
- 💾 All conversations automatically saved to SQLite database
- 📜 Chat history available anytime via /history endpoint
- 📄 Source documents returned with every answer
- ⚡ Fast API responses using FastAPI and Uvicorn

## 🛠️ Tech Stack
- **Backend:** FastAPI
- **LLM:** Groq API (Llama 3.3 70B)
- **AI Framework:** LangChain
- **Vector Database:** FAISS
- **Embeddings:** HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- **Database:** SQLite
- **PDF Loader:** PyPDFLoader
- **Server:** Uvicorn
- **Language:** Python 3.12

## 📁 Project Structure
```
MediBot/
├── data/                          # Medical PDF files
├── vectorstore/                   # FAISS vector database
├── create_memory_for_llm.py       # Loads PDF and creates FAISS database
├── connect_memory_with_llm.py     # Testing file for LLM connection
├── medibot.py                     # Main FastAPI application
├── requirements.txt               # Python dependencies
├── .env                           # API keys (not pushed to GitHub)
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```
## ⚙️ Installation & Setup

### Step 1 - Clone the Repository
```bash
git clone https://github.com/Panchasheelag89/medical-chatbot.git
cd medical-chatbot
```

### Step 2 - Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 - Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 - Create .env File
```bash
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### Step 5 - Add Medical PDF
```
Place your medical PDF file inside the data/ folder
```

### Step 6 - Create FAISS Database
```bash
python create_memory_for_llm.py
```

### Step 7 - Run the Application
```bash
python medibot.py
```

## 🔗 API Endpoints

### POST /chat
Accepts a medical question and returns an AI-generated answer with source documents.

**Request Body:**
```json
{
  "query": "What is diabetes?"
}
```

**Response:**
```json
{
  "result": "Diabetes mellitus is a chronic disease...",
  "source_documents": "..."
}
```

### GET /history
Returns all previous conversations stored in the SQLite database.

**Response:**
```json
{
  "history": [
    {
      "user_question": "What is diabetes?",
      "bot_answer": "Diabetes mellitus is a chronic disease...",
      "timestamp": "2026-02-28 20:02:06"
    }
  ]
}
```

