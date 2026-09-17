from pathlib import Path
import time
from loader import PDFLoader
from cleaner import TextCleaner
from chunker import TextChunker
from embedding import EmbeddingModel
from vector_store import VectorStore

def main():

    start_time = time.time()

    # ==========================================
    # Load PDFs
    # ==========================================

    print("Loading PDF documents...")

    loader = PDFLoader()

    documents = loader.load("data")

    print(f"Loaded {len(documents)} pages.")

    # ==========================================
    # Clean
    # ==========================================

    print("Cleaning documents...")

    cleaner = TextCleaner()

    documents = cleaner.clean(documents)

    print("Documents cleaned.")

    # ==========================================
    # Parent / Child Chunking
    # ==========================================

    print("Creating Parent/Child chunks...")

    chunker = TextChunker(
        parent_chunk_size=1500,
        parent_overlap=200,
        child_chunk_size=500,
        child_overlap=100
    )

    children = chunker.chunk(documents)

    parents = chunker.get_parents()

    print(f"Created {len(parents)} parent chunks.")

    print(f"Created {len(children)} child chunks.")

    # ==========================================
    # Embedding
    # ==========================================

    print("Loading embedding model...")

    embedding_model = (EmbeddingModel().get_embeddings())

    # ==========================================
    # FAISS
    # ==========================================

    print("Creating FAISS index...")

    vector_store = VectorStore.create(
        children,
        embedding_model
    )

    # ==========================================
    # Save
    # ==========================================

    Path("indexes").mkdir(exist_ok=True)

    VectorStore.save(
        vector_store,
        "indexes",
        parents
    )

    print("FAISS index saved.")

    print("Parent documents saved.")

    # ==========================================
    # Statistics
    # ==========================================

    print()
    print("=" * 60)

    print(f"Parents : {len(parents)}")

    print(f"Children: {len(children)}")

    print(
        f"Finished in "
        f"{time.time() - start_time:.2f} seconds."
    )

    print("=" * 60)


if __name__ == "__main__":

    main()