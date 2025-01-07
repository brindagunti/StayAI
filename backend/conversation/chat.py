from backend.memory.chroma_memory.retrieve_data import query_chroma
from backend.llms.groq_llm.inference import GroqInference

groq_llm = GroqInference()

def chat_with_travel_assistant():
    system_prompt = """
    You are a knowledgeable travel assistant specializing in Hyderabad, India. Your job is to answer 
    questions based on the Wikipedia data about Hyderabad that has been provided to you.
    
    With each user query, you will receive relevant sections from the Hyderabad Wikipedia article.
    Use this information to provide accurate and helpful answers about Hyderabad's history, culture,
    landmarks, cuisine, or any other aspects of the city.
    
    Instructions:
    1. Answer questions using only the information from the provided documents
    2. Keep answers brief and concise
    3. If you can't find specific information in the documents, politely say so
    4. Ask relevant follow-up questions to better understand what specific aspects of Hyderabad 
       the user is interested in
    5. If the user asks about something not covered in the Wikipedia article, acknowledge that 
       and stick to what you can confidently answer from the provided data
    """

    messages = [{"role": "system", "content": system_prompt}]

    while True:
        print("\n" + "=" * 80)
        user_query: str = input("\n🤔 Ask your question: ")
        print("\n" + "-" * 80)

        documents: str = query_chroma(
            user_query, collection_name="travel_data", n_results=3
        )
        print("\n📚 Knowledge Source:")
        print("-" * 80)
        print(f"\n{documents}\n")
        print("=" * 80)

        messages.append(
            {
                "role": "user",
                "content": f"""
            USER QUERY: {user_query}
            
            RELEVANT DOCUMENTS:
            {documents}
            """,
            }
        )

        assistant_answer: str = groq_llm.generate_response(messages)
        print("\n✨ Travel Assistant:")
        print("-" * 80)
        print(f"\n{assistant_answer}\n")
        print("=" * 80)

        messages.append({"role": "assistant", "content": assistant_answer})

