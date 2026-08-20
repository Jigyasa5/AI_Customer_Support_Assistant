import os

from pypdf import PdfReader
from docx import Document


def extract_pdf(path, category="general"):

    documents = []

    reader = PdfReader(path)

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            documents.append({

                "filename": os.path.basename(path),

                "page_number": page_number,

                "category": category,

                "document_type": "pdf",

                "text": text.strip()
            })

    return documents


def extract_txt(path, category="general"):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    return [{

        "filename": os.path.basename(path),

        "page_number": 1,

        "category": category,

        "document_type": "txt",

        "text": text.strip()
    }]


def extract_docx(path, category="general"):

    document = Document(path)

    paragraphs = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():

            paragraphs.append(
                paragraph.text.strip()
            )

    text = "\n".join(paragraphs)

    return [{

        "filename": os.path.basename(path),

        "page_number": 1,

        "category": category,

        "document_type": "docx",

        "text": text
    }]


def process_document(
    path,
    category="general"
):

    extension = os.path.splitext(
        path
    )[1].lower()

    if extension == ".pdf":

        return extract_pdf(
            path,
            category
        )

    elif extension == ".txt":

        return extract_txt(
            path,
            category
        )

    elif extension == ".docx":

        return extract_docx(
            path,
            category
        )

    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )


def process_document_folder(
    folder_path
):

    all_documents = []

    for filename in os.listdir(
        folder_path
    ):

        path = os.path.join(
            folder_path,
            filename
        )

        if os.path.isfile(path):

            try:

                documents = process_document(
                    path
                )

                all_documents.extend(
                    documents
                )

            except ValueError:
                pass

    return all_documents