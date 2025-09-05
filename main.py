from fastapi import FastAPI, Query
from pydantic import BaseModel
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model


app = FastAPI(title="askPESU API")

@app.on_event("startup")
def startup_event():
    global rag_chain
    qdrant_api_key = os.getenv("qdrant_api_key")
    gemini_api_key = os.getenv("gemini_api_key")

    client = QdrantClient(
        url=os.getenv("qdrant_url"),
        api_key=qdrant_api_key
    )

    embeddings = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-modernbert-base")

    vector_store = QdrantVectorStore(
       collection_name="reddit_vectors",
       embedding=embeddings,
       client=client
    )

    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai",google_api_key=gemini_api_key)

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(),
        llm=llm
    )

    prompt = ChatPromptTemplate.from_messages([
       ("system",
        "You are askPESU, developed by the PESU Dev team. "
        "Your sole purpose is to answer questions related to PES University. "
        "All your knowledge comes from the r/pesu subreddit. "
        "If someone asks 'who are you?' or 'introduce yourself', "
        "you must reply with: "
        "'I am askPESU, developed by the PESU Dev team. "
        "My sole purpose is to answer questions related to PES University, "
        "and my knowledge comes entirely from the r/pesu subreddit.'"),
       ("human",
        """You are an assistant for question-answering tasks. 
Use the following context to answer the question concisely and clearly. 
If you don't know the answer, say you don't know. 

Question: {question}
Context: {context}
Answer:""",
        ),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


@app.get("/ask")
def ask_pesu(query: str = Query(..., description="User question")):
    """Answer a question using PESU RAG pipeline."""
    try:
        answer = rag_chain.invoke(query)
        return {"query": query, "answer": answer}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/health")
def health():
    try:
        return "works"
    except Exception as e:
        return {"error": str(e)}
    

@app.get("/")
async def root():
    return {"message": "Hello! The server is running."}

if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run("main:app", host="0.0.0.0", port=7860)