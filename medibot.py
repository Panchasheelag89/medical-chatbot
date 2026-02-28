import os
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_FAISS_PATH="vectorstore/db_faiss"

def init_db():
    conn = sqlite3.connect("medibot.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_question TEXT,
            bot_answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_to_db(user_question, bot_answer):
    conn = sqlite3.connect("medibot.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (user_question, bot_answer) VALUES (?, ?)", (user_question, bot_answer))
    conn.commit()
    conn.close()

def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm

app = FastAPI()

class Request(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: Request):
    try:
        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """

        vectorstore=get_vectorstore()
        if vectorstore is None:
            return {"error": "Failed to load the vector store"}

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(
                model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.0,
                groq_api_key=os.environ["GROQ_API_KEY"],
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )

        response=qa_chain.invoke({'query':request.query})
        result=response["result"]
        source_documents=response["source_documents"]
        save_to_db(request.query, result)
        return {"result": result, "source_documents": str(source_documents)}

    except Exception as e:
        return {"error": str(e)}

@app.get("/history")
async def get_history():
    conn = sqlite3.connect("medibot.db")
    cursor = conn.cursor()
    cursor.execute("SELECT user_question, bot_answer, timestamp FROM chat_history ORDER BY timestamp ASC")
    rows = cursor.fetchall()
    conn.close()
    return {"history": [{"user_question": row[0], "bot_answer": row[1], "timestamp": row[2]} for row in rows]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)