import streamlit as st
from src.llm import LLM
from src.retriever import HybridRetriever
from src.rag_chain import RAGChain
from src.embedding import EmbeddingModel
from src.vector_store import VectorStore

# ==========================================
# Page Config
# ==========================================

st.set_page_config(

    page_title="Persian RAG",
    page_icon="📄",
    layout="wide"
)

# ==========================================
# Load RAG
# ==========================================

@st.cache_resource
def load_rag():

    print("1. Loading embedding model...")

    embedding_model = (EmbeddingModel().get_embeddings())

    print("2. Embedding loaded.")

    print("3. Loading FAISS...")

    vector_store = VectorStore.load(
        "indexes",
        embedding_model
    )

    print("4. FAISS loaded.")

    print("5. Creating Hybrid Retriever...")
    retriever = HybridRetriever(
        vector_store=vector_store,
        k=6,
        final_k=5,
        semantic_weight=0.6,
        keyword_weight=0.4
    )

    print("6. Retriever created.")

    print("7. Loading LLM...")

    llm = LLM().get_llm()

    print("8. LLM loaded.")

    print("9. Creating RAGChain...")

    generator = RAGChain(
        llm,
        retriever
    )

    print("10. RAGChain created.")

    return generator

generator = load_rag()

# ==========================================
# Chat History
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# ==========================================
# UI
# ==========================================

st.title("📄 Persian RAG")

# ==========================================
# Display History
# ==========================================

for message in (st.session_state.messages):

    with st.chat_message(message["role"]):

        st.write(message["content"])

# ==========================================
# User Input
# ==========================================

question = st.chat_input("سؤال خود را وارد کنید...")

# ==========================================
# Process
# ==========================================

if question:

    # --------------------------------------
    # User message
    # --------------------------------------
    with st.chat_message("user"):

        st.write(question)
    # --------------------------------------
    # Recent history
    # --------------------------------------
    recent_history = (st.session_state.messages)
    # --------------------------------------
    # Generate
    # --------------------------------------
    (
        answer,
        chunks,
        has_answer,
        standalone_question
    ) = generator.generate(

        question,
        recent_history
    )
    # --------------------------------------
    # Save user
    # --------------------------------------
    st.session_state.messages.append(

        {
            "role":
                "user",

            "content":
                question
        }
    )
    # --------------------------------------
    # Assistant
    # --------------------------------------
    with st.chat_message("assistant"):

        st.write(answer)
    # --------------------------------------
    # Save assistant
    # --------------------------------------
    st.session_state.messages.append(

        {
            "role":
                "assistant",

            "content":
                answer
        }
    )
    # ======================================
    # Retrieved Documents
    # ======================================
    if has_answer:

        with st.expander("Retrieved Context"):

            st.write("**Standalone Question:**")

            st.info(standalone_question)

            st.divider()

            for i, chunk in enumerate(chunks, start=1):

                st.markdown(f"### Parent {i}")
                # ----------------------------------
                # Hybrid
                # ----------------------------------
                if (chunk["hybrid_score"] is not None):

                    st.write(
                        f"**Hybrid Score:** "
                        f"`{chunk['hybrid_score']:.4f}`"
                    )
                # ----------------------------------
                # FAISS
                # ----------------------------------
                if (chunk["semantic_score"] is not None):

                    st.write(
                        f"**FAISS L2:** "
                        f"`{chunk['semantic_score']:.4f}`"
                    )
                # ----------------------------------
                # BM25
                # ----------------------------------
                if (chunk["bm25_score"] is not None):

                    st.write(
                        f"**BM25:** "
                        f"`{chunk['bm25_score']:.4f}`"
                    )
                # ----------------------------------
                # Parent text
                # ----------------------------------
                st.write(chunk["text"])
                # ----------------------------------
                # Metadata
                # ----------------------------------

                st.caption(
                    f"Page: "
                    f"{chunk['page']} | "

                    f"Source: "
                    f"{chunk['source']}"
                )

                st.divider()