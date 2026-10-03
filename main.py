from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated, TypedDict, List
import os
import shutil
from datetime import datetime

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

app = FastAPI(title="Multi-Document Research Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Global components
# -----------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:3b", temperature=0)

PERSIST_DIR = "./chroma_db"
DATA_DIR = "./data"

os.makedirs(DATA_DIR, exist_ok=True)

def get_vectorstore():
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# -----------------------------
# Tools
# -----------------------------
@tool
def search_documents(query: str) -> str:
    """Search across all uploaded research documents. Use this for any question about the content of the documents."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant information found in the documents."
    
    results = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        results.append(f"[Source: {source}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(results)

@tool
def list_documents() -> str:
    """List all currently uploaded documents."""
    files = os.listdir(DATA_DIR)
    if not files:
        return "No documents have been uploaded yet."
    return "Uploaded documents:\n" + "\n".join(f"- {f}" for f in files)

tools = [search_documents, list_documents]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

# -----------------------------
# Agent
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def agent(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

agent_app = workflow.compile()

SYSTEM_PROMPT = """You are a helpful Research Assistant.
You have access to tools that can search and list uploaded documents.

Rules:
- Always use the search_documents tool when the question is about the content of the documents.
- Cite the source file when possible.
- Be clear and structured in your answers.
- If no documents are uploaded or no relevant information is found, say so.
"""

# -----------------------------
# API Endpoints
# -----------------------------
class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Multi-Document Research Agent is running"}

@app.post("/ask")
def ask(payload: Question):
    result = agent_app.invoke({
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=payload.question)
        ]
    })
    
    final_message = result["messages"][-1]
    answer = final_message.content
    
    tool_calls = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"tool": tc["name"], "args": tc["args"]})
    
    return {
        "question": payload.question,
        "answer": answer,
        "tool_calls": tool_calls
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a PDF or TXT file to the knowledge base."""
    file_path = os.path.join(DATA_DIR, file.filename)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Load and index the document
    if file.filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = file.filename
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    
    return {
        "message": f"Successfully uploaded and indexed {file.filename}",
        "chunks": len(chunks)
    }

@app.get("/documents")
def list_docs():
    files = os.listdir(DATA_DIR)
    return {"documents": files}
