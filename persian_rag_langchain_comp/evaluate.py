import json

from src.embedding import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import HybridRetriever


def load_questions():

    with open(
        "evaluation_questions.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def main():

    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    # -----------------------------
    # Embedding
    # -----------------------------

    print("\nLoading embedding model...")

    embedding_model = (
        EmbeddingModel()
        .get_embeddings()
    )

    # -----------------------------
    # FAISS
    # -----------------------------

    print("Loading FAISS index...")

    vector_store = VectorStore.load(
        "indexes",
        embedding_model
    )

    # -----------------------------
    # Hybrid Retriever + Reranker
    # -----------------------------

    print("Loading Hybrid Retriever...")

    retriever = HybridRetriever(
        vector_store=vector_store,
        k=6,
        final_k=5
    )

    # -----------------------------
    # Questions
    # -----------------------------

    questions = load_questions()

    print(
        f"Loaded {len(questions)} evaluation questions."
    )

    print()

    # -----------------------------
    # Metrics
    # -----------------------------

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0

    # -----------------------------
    # Evaluation
    # -----------------------------

    for question_number, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        expected_pages = set(
            item["expected_pages"]
        )

        print("=" * 70)

        print(
            f"Question {question_number}:"
        )

        print(question)

        print(
            f"Expected pages: "
            f"{sorted(expected_pages)}"
        )

        print("-" * 70)

        # Retrieval

        documents = retriever.invoke(
            question
        )

        retrieved_pages = []

        for rank, document in enumerate(
            documents,
            start=1
        ):

            page = document.metadata.get(
                "page"
            )

            source = document.metadata.get(
                "source"
            )

            hybrid_score = document.metadata.get(
                "hybrid_score"
            )

            reranker_score = document.metadata.get(
                "reranker_score"
            )

            retrieved_pages.append(page)

            print(
                f"{rank}. "
                f"Page={page} | "
                f"Hybrid={hybrid_score} | "
                f"Reranker={reranker_score}"
            )

        # -----------------------------
        # Hit@1
        # -----------------------------

        if (
            len(retrieved_pages) >= 1
            and retrieved_pages[0]
            in expected_pages
        ):

            hit_at_1 += 1

        # -----------------------------
        # Hit@3
        # -----------------------------

        if any(
            page in expected_pages
            for page in retrieved_pages[:3]
        ):

            hit_at_3 += 1

        # -----------------------------
        # Hit@5
        # -----------------------------

        if any(
            page in expected_pages
            for page in retrieved_pages[:5]
        ):

            hit_at_5 += 1

        print()

    total = len(questions)

    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    print(
        f"Total Questions : {total}"
    )

    print(
        f"Hit@1           : "
        f"{hit_at_1 / total:.2%}"
    )

    print(
        f"Hit@3           : "
        f"{hit_at_3 / total:.2%}"
    )

    print(
        f"Hit@5           : "
        f"{hit_at_5 / total:.2%}"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()