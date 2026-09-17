import re
from langchain_core.documents import Document


class TextCleaner:
    """
    Cleans LangChain Document objects while preserving metadata.
    """

    def clean(self, documents: list[Document]) -> list[Document]:

        cleaned_documents = []

        for document in documents:

            text = document.page_content

            # -----------------------------
            # Normalize Persian characters
            # -----------------------------
            text = text.replace("ي", "ی")
            text = text.replace("ك", "ک")
            text = text.replace("تاریخ آ:خرین بازنگری", "")
            text = text.replace("تاریخ آخرین بازنگری", "")

            # -----------------------------
            # Remove repeated document headers
            # -----------------------------
            header_patterns = [

                r"شرکت\s*\n*\s*پالود\s*\n*\s*پارسیان\s*\n*\s*خیام",

                r"کد\s*سند.*",

                r"تاریخ\s*آخرین\s*بازنگری.*",

                r"شماره\s*آخرین\s*بازنگری.*",

                r"\d+\s*-\s*\d+"
            ]

            for pattern in header_patterns:

                text = re.sub(
                    pattern,
                    "",
                    text,
                    flags=re.IGNORECASE
                )

            # -----------------------------
            # Remove dotted table of contents
            # -----------------------------
            text = re.sub(
                r"\.{5,}",
                "",
                text
            )

            # -----------------------------
            # Remove lines containing only symbols
            # -----------------------------
            text = re.sub(
                r"^[\.\-\_\•\:\،\(\)\[\]]+$",
                "",
                text,
                flags=re.MULTILINE
            )

            # -----------------------------
            # Remove page numbers
            # -----------------------------
            text = re.sub(
                r"^\s*\d+\s*$",
                "",
                text,
                flags=re.MULTILINE
            )

            # -----------------------------
            # Remove very short lines
            # -----------------------------
            cleaned_lines = []

            for line in text.splitlines():

                line = line.strip()

                if len(line) <= 2:
                    continue

                cleaned_lines.append(line)

            text = "\n".join(cleaned_lines)

            # -----------------------------
            # Remove extra spaces
            # -----------------------------
            text = re.sub(
                r"[ \t]+",
                " ",
                text
            )

            # -----------------------------
            # Remove multiple blank lines
            # -----------------------------
            text = re.sub(
                r"\n\s*\n+",
                "\n\n",
                text
            )

            # -----------------------------
            # Fix hyphenated words
            # -----------------------------
            text = re.sub(
                r"(\w)-\n(\w)",
                r"\1\2",
                text
            )

            # Update page content
            document.page_content = text.strip()

            cleaned_documents.append(
                document
            )

        return cleaned_documents