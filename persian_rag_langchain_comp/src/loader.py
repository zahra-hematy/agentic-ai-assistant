from pathlib import Path
import fitz
import pytesseract
from langchain_core.documents import Document
import os

class PDFLoader:
   
    """
    Loads PDF documents and extracts their text using Tesseract OCR.

    Each PDF page is converted to an image and processed using Persian
    and English OCR. The extracted text is stored as a LangChain Document
    together with the source filename and page number as metadata.
    """

    def __init__(self):

            self.tesseract_cmd = (r"C:\Program Files\Tesseract-OCR\tesseract.exe")

            self.tessdata_dir = (r"C:\Program Files\Tesseract-OCR\tessdata")

            pytesseract.pytesseract.tesseract_cmd = (self.tesseract_cmd)

            os.environ["TESSDATA_PREFIX"] = (self.tessdata_dir)

    def load(self, folder_path: str):

        documents = []

        pdf_folder = Path(folder_path)

        pdf_files = list(pdf_folder.glob("*.pdf"))

        total_pdfs = len(pdf_files)

        for pdf_index, pdf_file in enumerate(pdf_files, start=1):

            print()
            print("=" * 70)
            print(f"PDF {pdf_index}/{total_pdfs}: {pdf_file.name}")
            print("=" * 70)

            doc = fitz.open(pdf_file)

            # Skip first two pages
            start_page = 2

            total_pages = len(doc)
            pages_to_process = total_pages - start_page

            for page_number, page in enumerate(
                doc,
                start=0
            ):

                if page_number < start_page:
                    continue

                current_page = page_number - start_page + 1

                progress = (
                    current_page / pages_to_process
                ) * 100

                print(
                    f"\rPage {page_number + 1}/{total_pages} "
                    f"| Progress: {progress:6.2f}% "
                    f"| OCR: {pdf_file.name}",
                    end="",
                    flush=True
                )

                # Convert PDF page to image
                pix = page.get_pixmap(
                    dpi=200,
                    alpha=False
                )

                image_path = "temp_page.png"

                pix.save(image_path)

                # OCR
                text = pytesseract.image_to_string(
                    image_path,
                    lang="fas+eng",
                   # config="--psm 6"
                )

                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": pdf_file.name,
                            "page": page_number + 1
                        }
                    )
                )

            print()
            print(
                f"Completed: {pdf_file.name}"
            )

            doc.close()

        print()
        print("=" * 70)
        print(
            f"OCR completed: {len(documents)} pages"
        )
        print("=" * 70)

        return documents
