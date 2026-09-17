from src.embedding import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import HybridRetriever
import json


# ----------------------------------
# Load Embedding
# ----------------------------------

print("1. Loading embedding model...")

embedding_model = EmbeddingModel().get_embeddings()

print("2. Embedding loaded.")


# ----------------------------------
# Load FAISS
# ----------------------------------

print("3. Loading FAISS...")

vector_store = VectorStore.load(
    "indexes",
    embedding_model
)

print("4. FAISS loaded.")


# ----------------------------------
# Load Evaluation Dataset
# ----------------------------------

with open(
    "evaluation_questions.json",
    "r",
    encoding="utf-8"
) as f:

    questions = json.load(f)

print(
    f"5. Loaded {len(questions)} evaluation questions."
)


# ----------------------------------
# Weight Evolution
# ----------------------------------

weights = [

    (1.0, 0.0),
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9),
    (0.0, 1.0),

]


for semantic_weight, keyword_weight in weights:

    print("\n" + "=" * 70)

    print(
        f"Testing weights: "
        f"FAISS={semantic_weight} | "
        f"BM25={keyword_weight}"
    )

    retriever = HybridRetriever(

        vector_store=vector_store,

        k=5,
        final_k=3,

        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight

    )

    hit1 = 0
    hit3 = 0

    for item in questions:

        query = item["question"]
        expected_pages = item["expected_pages"]

        results = retriever.invoke(query)

        pages = [
            doc.metadata["page"]
            for doc in results
        ]

        if any(
            page in expected_pages
            for page in pages[:1]
        ):
            hit1 += 1

        if any(
            page in expected_pages
            for page in pages[:3]
        ):
            hit3 += 1

    total = len(questions)

    print(
        f"Hit@1 = {hit1 / total * 100:.2f}%"
    )

    print(
        f"Hit@3 = {hit3 / total * 100:.2f}%"
    )