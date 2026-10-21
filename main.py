from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated, TypedDict
import os
import shutil

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
# Configuration
# -----------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:3b", temperature=0)

PERSIST_DIR = "./chroma_db"
DATA_DIR = "./data"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def get_vectorstore():
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# -----------------------------
# Tools
# -----------------------------
@tool
def search_documents(query: str) -> str:
    """Search across all uploaded documents. 
    You MUST use this tool for any question about the content of the documents.
    Always include the source filename in your final answer."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the uploaded documents."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        results.append(f"[Source: {source}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(results)

@tool
def list_documents() -> str:
    """List all currently uploaded documents."""
    files = [f for f in os.listdir(DATA_DIR) if not f.startswith(".")]
    if not files:
        return "No documents have been uploaded yet."
    return "Currently uploaded documents:\n" + "\n".join(f"- {f}" for f in files)

tools = [search_documents, list_documents]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

# -----------------------------
# Agent
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

agent_app = workflow.compile()

SYSTEM_PROMPT = """You are a precise Research Assistant.

Rules:
1. For ANY question about the content of the documents → ALWAYS use the search_documents tool first.
2. Use list_documents when the user asks what files are available.
3. In your final answer, clearly cite the source filename(s) you used.
4. Be structured, concise, and factual.
5. If no relevant information is found, say so honestly.
"""

# -----------------------------
# API Endpoints
# -----------------------------
class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Multi-Document Research Agent is running", "version": "1.2"}

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
    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

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
        "message": f"Successfully uploaded and indexed '{file.filename}'",
        "chunks_added": len(chunks)
    }

@app.get("/documents")
def list_docs():
    files = [f for f in os.listdir(DATA_DIR) if not f.startswith(".")]
    return {"documents": files}

@app.delete("/clear")
def clear_knowledge_base():
    """Clear all uploaded documents and reset the vector store."""
    # Remove all files in data/
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Reset Chroma
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    os.makedirs(PERSIST_DIR, exist_ok=True)

    return {"message": "Knowledge base cleared successfully. All documents and embeddings removed."}
