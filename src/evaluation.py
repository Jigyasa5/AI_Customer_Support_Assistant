import pandas as pd
from sklearn.metrics import accuracy_score


# Evaluation Dataset

def create_evaluation_dataset():

    data = [

        {
            "question": "How can I get a refund?",
            "expected_answer": "Customers can request a refund within 30 days of receiving their order.",
            "expected_source": "Refund Policy"
        },

        {
            "question": "How long does a refund take?",
            "expected_answer": "Refunds are generally processed within 5-7 business days.",
            "expected_source": "Refund Policy"
        },

        {
            "question": "How can I track my order?",
            "expected_answer": "Customers can track their order using the tracking number provided in the order confirmation email.",
            "expected_source": "Order Tracking"
        },

        {
            "question": "What payment methods are accepted?",
            "expected_answer": "Credit cards, debit cards and online payment methods are accepted.",
            "expected_source": "Payment"
        },

        {
            "question": "What happens if my payment fails?",
            "expected_answer": "Customers can retry the payment or use another payment method.",
            "expected_source": "Payment"
        },

        {
            "question": "Can I cancel my order?",
            "expected_answer": "Orders can be cancelled before they are shipped.",
            "expected_source": "Order Cancellation"
        },

        {
            "question": "How long does standard delivery take?",
            "expected_answer": "Standard delivery usually takes 3-5 business days.",
            "expected_source": "Delivery"
        },

        {
            "question": "Do you provide international shipping?",
            "expected_answer": "I couldn't find this information in the available knowledge base.",
            "expected_source": "None"
        }

    ]

    return pd.DataFrame(data)


# Check Answer

def check_answer(
    generated_answer,
    expected_answer
):

    generated_answer = generated_answer.lower()
    expected_answer = expected_answer.lower()

    # Check whether important words
    # from expected answer appear
    expected_words = set(
        expected_answer.split()
    )

    generated_words = set(
        generated_answer.split()
    )

    if not expected_words:
        return 0

    overlap = (
        expected_words
        & generated_words
    )

    score = len(overlap) / len(
        expected_words
    )

    return score


# Run Evaluation

def evaluate_rag(
    evaluation_df,
    generate_function,
    collection,
    embedding_model,
    generator
):

    results = []

    for _, row in evaluation_df.iterrows():

        question = row["question"]

        expected_answer = row[
            "expected_answer"
        ]

        generated_answer = generate_function(
            question,
            collection,
            embedding_model,
            generator
        )

        score = check_answer(
            generated_answer,
            expected_answer
        )

        results.append({

            "question": question,

            "expected_answer":
                expected_answer,

            "generated_answer":
                generated_answer,

            "score":
                score
        })

    return pd.DataFrame(results)

def evaluate_retrieval(
    evaluation_df,
    collection,
    embedding_model,
    top_k=3
):

    results = []

    for _, row in evaluation_df.iterrows():

        question = row["question"]

        expected_source = row[
            "expected_source"
        ]

        query_embedding = embedding_model.encode(
            [question]
        ).tolist()

        search_results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        documents = search_results[
            "documents"
        ][0]

        # Check whether expected source/topic
        # appears in retrieved documents
        found = False

        if expected_source != "None":

            for document in documents:

                if expected_source.lower() in document.lower():

                    found = True
                    break

        results.append({

            "question": question,

            "expected_source":
                expected_source,

            "retrieved":
                found
        })

    return pd.DataFrame(results)


# Calculate Overall Score

def calculate_average_score(
    results_df
):

    average_score = results_df[
        "score"
    ].mean()

    return average_score