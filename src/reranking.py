from sentence_transformers import CrossEncoder


# Load reranking model

def load_reranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
):

    reranker = CrossEncoder(
        model_name
    )

    return reranker


# ------------------Rerank retrieved documents -----------------


def rerank_documents(
    query,
    documents,
    reranker,
    top_k=3
):

    # Create query-document pairs
    pairs = [
        [query, document]
        for document in documents
    ]

    # Get relevance scores
    scores = reranker.predict(
        pairs
    )

    # Combine documents and scores
    results = []

    for document, score in zip(
        documents,
        scores
    ):

        results.append({
            "document": document,
            "score": float(score)
        })

    # Sort by relevance
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]