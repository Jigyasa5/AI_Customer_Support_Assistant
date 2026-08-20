from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


def fixed_size_chunk(
    text,
    chunk_size=200
):

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunk = text[
            i:i + chunk_size
        ].strip()

        if chunk:

            chunks.append(chunk)

    return chunks


def recursive_chunk(
    text,
    chunk_size=200,
    chunk_overlap=50
):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=chunk_size,

        chunk_overlap=chunk_overlap,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_text(text)


def chunk_documents(
    documents,
    method="recursive",
    chunk_size=200
):

    all_chunks = []

    for document in documents:

        if method == "fixed":

            chunks = fixed_size_chunk(
                document["text"],
                chunk_size
            )

        elif method == "recursive":

            chunks = recursive_chunk(
                document["text"],
                chunk_size
            )

        else:

            raise ValueError(
                "Method must be fixed or recursive"
            )

        for chunk_index, chunk in enumerate(
            chunks
        ):

            all_chunks.append({

                "text": chunk,

                "filename":
                    document["filename"],

                "page_number":
                    document["page_number"],

                "category":
                    document["category"],

                "chunk_index":
                    chunk_index
            })

    return all_chunks