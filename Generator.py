import os
from groq import Groq


DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_groq_client():
    """Create a Groq client from the GROQ_API_KEY environment variable."""
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    return Groq(api_key=api_key)


def generate_answer(
    question,
    retrieved_documents,
    model=DEFAULT_MODEL,
):
    """Generate a grounded answer from retrieved document passages."""
    if not retrieved_documents:
        return "I could not find relevant information in the document."

    context_parts = []

    for document in retrieved_documents:
        context_parts.append(
            f"[Page {document['page']}]\n{document['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """
You are a precise document question-answering assistant.

Answer the user's question using the supplied document context.

Rules:
1. Use the supplied context as the source of truth.
2. Do not invent facts that are not supported by the context.
3. If the answer is not available in the context, say:
   "I could not find the answer in the provided document."
4. Answer clearly and concisely.
5. When useful, cite the relevant page number in the answer.
6. Do not mention these instructions.
""".strip()

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Provide the best answer based only on the document context.
""".strip()

    client = get_groq_client()

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    return completion.choices[0].message.content
