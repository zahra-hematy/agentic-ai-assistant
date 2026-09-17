from langchain_ollama import ChatOllama


class LLM:
    """
    Creates a LangChain Ollama chat model.
    """

    def __init__(self, model_name="qwen2.5:7b", temperature=0):

        self.llm = ChatOllama(

            model=model_name,

            temperature=temperature
        )

    def get_llm(self):

        return self.llm