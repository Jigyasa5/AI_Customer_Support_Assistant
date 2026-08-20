# Hallucination Handling

FALLBACK_MESSAGE = (
    "I couldn't find this information in the "
    "available knowledge base."
)


# Check whether relevant context exists

def has_relevant_context(
    retrieved_documents,
    max_distance=1.5
):

    if not retrieved_documents:
        return False

    for item in retrieved_documents:

        distance = item.get(
            "distance",
            999
        )

        if distance <= max_distance:
            return True

    return False


# Get safe context

def get_safe_context(
    retrieved_documents,
    max_distance=1.5
):

    if not has_relevant_context(
        retrieved_documents,
        max_distance
    ):

        return None

    context = "\n\n".join(
        item["document"]
        for item in retrieved_documents
    )

    return context


# Generate fallback response

def hallucination_safe_response(
    retrieved_documents,
    max_distance=1.5
):

    context = get_safe_context(
        retrieved_documents,
        max_distance
    )

    if context is None:

        return FALLBACK_MESSAGE

    return context