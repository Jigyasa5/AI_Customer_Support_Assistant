# Retrieve documents with their source information


def retrieve_with_sources(
    query,
    collection,
    embedding_model,
    top_k=3
):

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        retrieved.append({

            "document": document,

            "filename": metadata.get(
                "filename",
                "Unknown"
            ),

            "page_number": metadata.get(
                "page_number",
                "Unknown"
            ),

            "category": metadata.get(
                "category",
                "Unknown"
            ),

            "chunk_index": metadata.get(
                "chunk_index",
                "Unknown"
            ),

            "distance": distance
        })

    return retrieved


#------------------ Create context with source information -----------------

def create_context_with_sources(
    retrieved_documents
):

    context_parts = []

    for item in retrieved_documents:

        context_parts.append(
            f"""
SOURCE:
{item['filename']} — Page {item['page_number']}

CONTENT:
{item['document']}
"""
        )

    return "\n".join(context_parts)


#------------- Display sources -----------------

def display_sources(
    retrieved_documents
):

    print("\nSOURCES: ")
    for i, item in enumerate(
        retrieved_documents
    ):

        print(
            f"{i + 1}. "
            f"{item['filename']} — "
            f"Page {item['page_number']}"
        )