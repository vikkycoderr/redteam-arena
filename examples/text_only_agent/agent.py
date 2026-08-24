import ollama


def answer(question: str):
    return ollama.chat(
        model="example-local-model",
        messages=[
            {"role": "system", "content": "Answer text questions."},
            {"role": "user", "content": question},
        ],
    )
