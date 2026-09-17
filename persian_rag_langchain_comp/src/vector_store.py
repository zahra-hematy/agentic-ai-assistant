from langchain_community.vectorstores import FAISS
from pathlib import Path
import pickle

class VectorStore:
    """
    Creates, saves and loads FAISS vector store
    together with Parent documents.
    """
    # ==============================================
    # Create
    # ==============================================

    @staticmethod
    def create(documents, embedding_model):

        return FAISS.from_documents(documents, embedding_model)

    # ==============================================
    # Save
    # ==============================================

    @staticmethod
    def save(vector_store, folder_path, parents):

        folder = Path(folder_path)

        folder.mkdir(exist_ok=True)

        # ------------------------------------------
        # Save FAISS
        # ------------------------------------------

        vector_store.save_local(str(folder))

        # ------------------------------------------
        # Save Parents
        # ------------------------------------------

        parent_path = (folder / "parents.pkl")

        with open(parent_path, "wb") as file:

            pickle.dump(parents, file)

    # ==============================================
    # Load
    # ==============================================

    @staticmethod
    def load(folder_path, embedding_model):

        folder = Path(folder_path)

        # ------------------------------------------
        # Load FAISS
        # ------------------------------------------

        vector_store = FAISS.load_local(
            str(folder),
            embedding_model,
            allow_dangerous_deserialization=True
        )

        # ------------------------------------------
        # Load Parents
        # ------------------------------------------

        parent_path = (folder / "parents.pkl")

        if not parent_path.exists():

            raise FileNotFoundError("parents.pkl was not found.")

        with open(parent_path, "rb") as file:

            parents = pickle.load(file)

        # Attach parents to FAISS
        vector_store.parent_documents = parents

        return vector_store