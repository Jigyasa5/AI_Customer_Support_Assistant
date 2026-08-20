from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_embedding_models():
    models = {
        "MiniLM": SentenceTransformer(
            "all-MiniLM-L6-v2"
        ),
        "MPNet": SentenceTransformer(
            "all-mpnet-base-v2"
        )
    }

    return models

def generate_embeddings(model,texts):
    embeddings = model.encode(texts,show_progress_bar=True)
    return embeddings

def calculate_similarity(model,sentences):

    vectors = model.encode(sentences)
    similarity = cosine_similarity(vectors)
    return similarity

def compare_models():
    models = load_embedding_models()
    sentence = "I want a refund for my order"

    for name, model in models.items():
        embedding = model.encode(sentence)

        print(name,"embedding shape:",embedding.shape)


# Test Model Similarity

def compare_similarity():
    models = load_embedding_models()
    sentences = [
        "I want my money back",
        "I need a refund",
        "Where is my order?"
    ]

    for name, model in models.items():
        similarity = calculate_similarity(model,sentences)

        print(f"\n{name} similarity:")
        print(similarity)