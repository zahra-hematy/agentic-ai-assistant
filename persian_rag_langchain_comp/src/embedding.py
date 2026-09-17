from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:
    """
    Creates embedding vectors using HuggingFace models.
    """

    def __init__(self, model_name="intfloat/multilingual-e5-small"):

        self.model = HuggingFaceEmbeddings(
            model_name=model_name
        )

    def get_embeddings(self):

        return self.model