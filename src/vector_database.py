import chromadb

from sentence_transformers import SentenceTransformer


def create_chroma_client(
    path="/chroma_db"
):

    client = chromadb.PersistentClient(
        path=path
    )

    return client


def create_collection(
    client,
    name="customer_support"
):

    try:

        collection = client.get_collection(
            name=name
        )

    except Exception:

        collection = client.create_collection(
            name=name
        )

    return collection


def add_chunks(
    collection,
    chunks,
    embedding_model
):

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [

        {
            "filename": chunk["filename"],
            "page_number": str(
                chunk["page_number"]
            ),
            "category": chunk["category"],
            "chunk_index": str(
                chunk["chunk_index"]
            )
        }

        for chunk in chunks
    ]

    collection.add(

        ids=ids,

        documents=documents,

        embeddings=embeddings,

        metadatas=metadatas
    )

    print(
        f"Added {len(chunks)} chunks to ChromaDB."
    )


def search_collection(
    collection,
    query,
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

    return results