from transformers import pipeline


def load_generator():

    generator = pipeline(

        "text-generation",

        model="gpt-oss-20b",

        max_new_tokens=100,

        do_sample=False
    )

    return generator


def retrieve_context(
    query,
    collection,
    embedding_model,
    top_k=3
):

    results = collection.query(

        query_embeddings=
        embedding_model.encode(
            [query]
        ).tolist(),

        n_results=top_k
    )

    documents = results["documents"][0]

    context = "\n\n".join(
        documents
    )

    return context


def create_prompt(
    query,
    context
):

    return f"""
You are an AI customer support assistant.

IMPORTANT RULES:

1. Answer ONLY using the CONTEXT.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. Do NOT make up information.
5. If the answer is not present in the context, say exactly:

I don't have enough information to answer this question.

Keep the answer concise.

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""


def generate_answer(
    query,
    collection,
    embedding_model,
    generator,
    top_k=3
):

    context = retrieve_context(

        query,

        collection,

        embedding_model,

        top_k
    )

    prompt = create_prompt(
        query,
        context
    )

    response = generator(
        prompt
    )

    generated_text = response[0][
        "generated_text"
    ]

    if "ANSWER:" in generated_text:

        answer = generated_text.split(
            "ANSWER:"
        )[-1].strip()

    else:

        answer = generated_text.strip()

    return answer