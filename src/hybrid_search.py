from rank_bm25 import BM25Okapi
import numpy as np


#---------- 1. Create BM25 index from ChromaDB documents-----------

def create_bm25_index(collection):

    results = collection.get(
        include=["documents"]
    )

    documents = results["documents"]

    tokenized_documents = [
        document.lower().split()
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    return bm25, documents


# 2. BM25 Keyword Search

def bm25_search(
    query,
    bm25,
    documents,
    top_k=3
):

    query_tokens = query.lower().split()

    scores = bm25.get_scores(
        query_tokens
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "document": documents[index],
            "score": float(scores[index])
        })

    return results


#---------- 3. Vector Search   ----------------

def vector_search(
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
    distances = results["distances"][0]

    results_list = []

    for document, distance in zip(
        documents,
        distances
    ):

        results_list.append({
            "document": document,
            "distance": float(distance)
        })

    return results_list


#---------- 4. Hybrid Search   ----------------

def hybrid_search(
    query,
    collection,
    embedding_model,
    bm25,
    documents,
    top_k=3,
    alpha=0.5
):

    # Vector results
    vector_results = vector_search(
        query,
        collection,
        embedding_model,
        top_k
    )

    # BM25 results
    keyword_results = bm25_search(
        query,
        bm25,
        documents,
        top_k
    )

    combined_scores = {}

    # Add vector scores

    for result in vector_results:

        document = result["document"]

        # Convert distance into similarity
        similarity = 1 / (
            1 + result["distance"]
        )

        combined_scores[document] = (
            combined_scores.get(
                document,
                0
            )
            + alpha * similarity
        )

    # Add BM25 scores

    for result in keyword_results:

        document = result["document"]

        score = result["score"]

        combined_scores[document] = (
            combined_scores.get(
                document,
                0
            )
            + (1 - alpha) * score
        )

    # Sort results

    sorted_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {
            "document": document,
            "score": score
        }
        for document, score
        in sorted_results[:top_k]
    ]