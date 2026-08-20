
def search_with_metadata(
    collection,
    query,
    embedding_model,
    top_k=3,
    category=None,
    filename=None,
    page_number=None
):

    # Generate query embedding
    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    # Build metadata filter
    conditions = []

    if category is not None:
        conditions.append({
            "category": category
        })

    if filename is not None:
        conditions.append({
            "filename": filename
        })

    if page_number is not None:
        conditions.append({
            "page_number": str(page_number)
        })

    # Create ChromaDB filter
    where = None

    if len(conditions) == 1:

        where = conditions[0]

    elif len(conditions) > 1:

        where = {
            "$and": conditions
        }

    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where
    )

    return results



# Display filtered results

def display_filtered_results(results):

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    for i, document in enumerate(documents):

        print(f"\nResult {i + 1}")

        print("Document:")
        print(document)

        if i < len(metadatas):

            print(
                "Metadata:",
                metadatas[i]
            )

        if i < len(distances):

            print(
                "Distance:",
                distances[i]
            )

        print("-" * 80)