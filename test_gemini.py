from src.rag import load_generator


client = load_generator()


response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Say hello in one sentence."
)


print(response.text)