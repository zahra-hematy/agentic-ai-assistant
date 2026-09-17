from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

class TextChunker:
    """
    Creates hierarchical Parent/Child chunks.

    Parent:
        Larger chunk containing more context.

    Child:
        Smaller chunk used for retrieval.
    """

    def __init__(
        self,
        parent_chunk_size=1500,
        parent_overlap=200,
        child_chunk_size=500,
        child_overlap=100
    ):

        # ==========================================
        # Parent Splitter
        # ==========================================

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,

            separators=[
                "\n\n",
                "\n",
                "؟",
                "!",
                ".",
                " "
            ],

            keep_separator=True
        )

        # ==========================================
        # Child Splitter
        # ==========================================

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,

            separators=[
                "\n\n",
                "\n",
                "؟",
                "!",
                ".",
                " "
            ],

            keep_separator=True
        )

        # Parent documents
        self.parents = {}

    # ==============================================
    # Create Parent + Child
    # ==============================================

    def chunk(
        self,
        documents: list[Document]
    ) -> list[Document]:

        child_documents = []

        for document in documents:

            # --------------------------------------
            # Create Parent Chunks
            # --------------------------------------

            parent_chunks = self.parent_splitter.split_documents([document])

            for parent_index, parent in enumerate(parent_chunks):

                parent_id = str(uuid.uuid4())

                # ----------------------------------
                # Save Parent
                # ----------------------------------

                parent_document = Document(
                    page_content=parent.page_content,

                    metadata={
                        **parent.metadata,
                        "parent_id": parent_id,
                        "parent_index": parent_index
                    }
                )

                self.parents[parent_id] = parent_document

                # ----------------------------------
                # Create Child Chunks
                # ----------------------------------

                child_chunks = self.child_splitter.split_documents([parent])

                # ----------------------------------
                # Save Children
                # ----------------------------------

                for child_index, child in enumerate(child_chunks):

                    child_document = Document(
                        page_content=child.page_content,

                        metadata={
                            **child.metadata,
                            "parent_id": parent_id,
                            "parent_index": parent_index,
                            "child_index": child_index
                        }
                    )

                    child_documents.append(child_document)

        return child_documents

    # ==============================================
    # Get Parents
    # ==============================================

    def get_parents(self):

        return self.parents