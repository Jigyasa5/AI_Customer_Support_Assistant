from src.memory import format_history


def rewrite_query(
    query,
    generator
):

    history = format_history()

    if not history:

        return query

    prompt = f"""
Rewrite the latest user question into
a standalone search query.

Use the conversation history to understand
what the user is referring to.

Return ONLY the rewritten search query.
Do NOT answer the question.
Do NOT explain anything.

Conversation History:
{history}

Latest User Question:
{query}

Standalone Search Query:
"""

    response = generator(
        prompt
    )

    generated_text = response[0][
        "generated_text"
    ]

    # Extract text after the instruction
    if "Standalone Search Query:" in generated_text:

        rewritten = generated_text.split(
            "Standalone Search Query:"
        )[-1].strip()

    else:

        rewritten = query

    # Keep first line only
    rewritten = rewritten.split(
        "\n"
    )[0].strip()

    return rewritten