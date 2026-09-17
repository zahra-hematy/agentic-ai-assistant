from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class RAGChain:
    """
    Conversational RAG.
    Chat history is used only to rewrite
    follow-up questions into standalone questions.
    """

    def __init__(self, llm, retriever):

        self.llm = llm

        self.retriever = retriever

        # ==========================================
        # Question Rewriting
        # ==========================================

        self.rewrite_prompt = (
            ChatPromptTemplate.from_template(
                """
                You are a question rewriting assistant for a Persian
                document retrieval system.

                Rewrite the latest user question into a complete,
                standalone search question.

                Use the conversation history to understand references such as:

                - این
                - آن
                - این روش
                - این مرحله
                - چطور
                - چگونه
                - مراحلش
                - بیشتر توضیح بده

                Rules:

                1. Return ONLY the rewritten question.
                2. Do NOT answer the question.
                3. Do NOT invent information.
                4. Preserve important technical terms.
                5. Use both user questions and assistant answers.
                6. If the latest question is already standalone,
                return it unchanged.
                7. Answer in Persian.

                Conversation history:

                {chat_history}

                Latest user question:

                {question}

                Standalone search question:
                """
            )
        )

        # ==========================================
        # Answer Prompt
        # ==========================================

        self.answer_prompt = (
            ChatPromptTemplate.from_template(
                """
                You are a Persian document assistant.
                Use ONLY the provided context.
                If the answer exists:

                - Answer only in Persian.
                - Summarize the information.
                - Do not copy the document word by word.

                If the answer does not exist, reply exactly:

                اطلاعات کافی در اسناد موجود نیست.

                Context:

                {context}

                Question:

                {input}
                """
            )
        )

        # ==========================================
        # Document Chain
        # ==========================================

        document_chain = (
            create_stuff_documents_chain(
                self.llm,
                self.answer_prompt
            )
        )

        # ==========================================
        # Retrieval Chain
        # ==========================================

        self.chain = (
            create_retrieval_chain(
                self.retriever,
                document_chain
            )
        )

    # ==============================================
    # Rewrite Question
    # ==============================================

    def rewrite_question(self, question, chat_history):

        if not chat_history:

            return question

        history_text = ""

        for message in chat_history:

            role = message["role"]

            content = message["content"]

            if role == "user":

                history_text += (f"User: {content}\n")

            elif role == "assistant":

                history_text += (f"Assistant: {content}\n")

        prompt = (
            self.rewrite_prompt.invoke(
                {
                    "chat_history":
                        history_text,

                    "question":
                        question
                }
            )
        )

        response = self.llm.invoke(prompt)

        return response.content.strip()

    # ==============================================
    # Generate
    # ==============================================

    def generate(self, question, chat_history):
        # ------------------------------------------
        # Rewrite
        # ------------------------------------------

        standalone_question = (
            self.rewrite_question(
                question,
                chat_history
            )
        )

        print()
        print("=" * 70)
        print("Original Question:")
        print(question)
        print("-" * 70)
        print("Standalone Question:")
        print(standalone_question)
        print("=" * 70)

        # ------------------------------------------
        # Retrieval + Generation
        # ------------------------------------------

        response = (
            self.chain.invoke(
                {
                    "input":
                        standalone_question
                }
            )
        )

        # ------------------------------------------
        # Debug
        # ------------------------------------------

        print()
        print("=" * 70)
        print("RETRIEVED PARENT DOCUMENTS")
        print("=" * 70)

        for i, doc in enumerate(response["context"], start=1):

            print(f"\n--- Parent {i} ---")
            print("Source:", doc.metadata.get("source"))
            print("Page:", doc.metadata.get("page"))
            print("Hybrid:", doc.metadata.get("hybrid_score"))
            print(doc.page_content)

        answer = response["answer"]

        # ------------------------------------------
        # Check answer
        # ------------------------------------------

        no_answer_text = ("اطلاعات کافی در اسناد موجود نیست.")

        has_answer = (no_answer_text not in answer.strip())

        # ------------------------------------------
        # Retrieved chunks
        # ------------------------------------------

        chunks = []

        for doc in response["context"]:

            chunks.append(
                {
                    "text":
                        doc.page_content,
                    "page":
                        doc.metadata.get("page"),
                    "source":
                        doc.metadata.get("source"),
                    "semantic_score":
                        doc.metadata.get("semantic_score"),
                    "bm25_score":
                        doc.metadata.get("bm25_score"),
                    "hybrid_score":
                        doc.metadata.get("hybrid_score")
                }
            )

        return (
            answer,
            chunks,
            has_answer,
            standalone_question
        )