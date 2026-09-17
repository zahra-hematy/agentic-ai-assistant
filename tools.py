from langchain_core.tools import tool
from persian_rag_langchain_comp.src.embedding import EmbeddingModel
from persian_rag_langchain_comp.src.retriever import HybridRetriever
from persian_rag_langchain_comp.src.vector_store import VectorStore
import os
import smtplib
from dotenv import load_dotenv
from email.message import EmailMessage
from email_service import send_email_smtp
from scheduler import schedule_email_job
from time_parser import parse_persian_datetime

@tool
def calculator(a: float, b: float, operation: str) -> float:
    """Perform a basic mathematical operation."""

    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero.")

        return a / b

    raise ValueError("Unsupported operation.")


@tool
def get_customer_debt(customer_name: str) -> float:
    """Return the debt of a customer."""

    customers = {
        "علی": 10_000_000,
        "رضا": 20_000_000,
        "مریم": 15_000_000
    }

    return customers.get(customer_name, -1)


# ==========================================
# Load RAG components
# ==========================================

print("Loading embedding model...")

embedding_model = EmbeddingModel().get_embeddings()

print("Embedding loaded.")

print("Loading FAISS...")

vector_store = VectorStore.load(
    "persian_rag_langchain/indexes",
    embedding_model
)

print("FAISS loaded.")

print("Creating Hybrid Retriever...")

retriever = HybridRetriever(
    vector_store=vector_store,
    k=6,
    final_k=5,
    semantic_weight=0.6,
    keyword_weight=0.4
)

print("Hybrid Retriever created.")


# ==========================================
# RAG Tool
# ==========================================

@tool
def search_persian_docs(query: str) -> str:
    """
    Search the Persian company documents for information.

    Use this tool whenever the user's question is related to
    company procedures, warehouse, production, employees,
    responsibilities, instructions, or any information that
    may exist in the documents.

    Even if the user's question is short, incomplete, or
    ambiguous, try searching the documents first instead of
    asking for clarification.
    """
    print("========== RAG TOOL CALLED ==========")
    # print("QUERY:", query)

    documents = retriever.invoke(query)
    # print("DOCUMENTS FOUND:", len(documents))

    if not documents:

        return "اطلاعات کافی در اسناد موجود نیست."

    results = []
    print("DOCUMENTS FOUND:", len(documents))
    for i, doc in enumerate(documents, start=1):

        text = doc.page_content

        source = doc.metadata.get("source", "Unknown")

        page = doc.metadata.get("page", "Unknown")
    
        results.append(
            f"""
        --- Document {i} ---

        Source: {source}
        Page: {page}

        Content:
        {text}
        """
                )

    return "\n".join(results)


@tool
def send_email(
    to: str,
    subject: str,
    body: str
) -> str:
    """
    Send an email immediately.
    """

    return send_email_smtp(
        to=to,
        subject=subject,
        body=body
    )


@tool
def schedule_email(
    to: str,
    subject: str,
    body: str,
    send_at: str
) -> str:
    """
    Schedule an email for a future time.

    IMPORTANT:
    send_at must be an ISO-8601 datetime with timezone.

    Example:
    2026-09-16T09:00:00+03:30
    """

    return schedule_email_job(
        to=to,
        subject=subject,
        body=body,
        send_at=send_at
    )