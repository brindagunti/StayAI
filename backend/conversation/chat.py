from typing import List, Dict

from concurrent.futures import ThreadPoolExecutor
from backend.memory.chroma_memory.retrieve_data import query_chroma
from backend.llms.groq_llm.inference import GroqInference
from backend.memory.mem0_memory.try_mem0 import (
    extract_relevant_memories,
    add_memory_in_mem0,
)
from backend.utils.json_utils import (
    pre_process_the_json_response,
    load_object_from_string,
)
from backend.agents.simple_agent_framework.browser_agent import BrowserAgent
from threading import Thread

groq_llm = GroqInference()


def print_section(title: str = "", content: str = "", separator: str = "=") -> None:
    print(f"\n{separator * 80}")
    if title:
        print(f"\n{title}")
        print(f"{'-' * 80}")
    if content:
        print(f"\n{content}\n")
    if separator == "=":
        print(f"{separator * 80}")


def chat_with_travel_assistant(user_id: str, user_query: str, messages: List[Dict[str, str]]):

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
    agent = BrowserAgent()
    print_section()

    memories: list[str] = extract_relevant_memories(user_query, user_id)
    memories: list[str] = extract_relevant_memories(user_query, user_id)
    print_section("📚 Memories:", memories or "No relevant memories found.")

    rephrased_query: str = rephrase_user_query(user_query, memories)
    print_section("📚 Rephrased Query:", rephrased_query)

    documents: str = query_chroma(
        rephrased_query, collection_name="travel_data", n_results=3
    )

    print_section("📚 Knowledge Source:", documents)

    agent_response = agent.run(user_query)

    print_section("✨ Agent Response:", agent_response)

    messages.append(
        {
            "role": "user",
            "content": f"""
        USER QUERY: {user_query}
        
        RELEVANT MEMORIES:
        {memories}
        
        RELEVANT DOCUMENTS:
        {documents}
        
        OBSERVATIONS FROM THE AGENT WHICH SURFED THE INTERNET:
        {agent_response}
        """,
        }
    )

    assistant_answer: str = groq_llm.generate_response(messages)
    messages.append({"role": "assistant", "content": assistant_answer})
    
    with ThreadPoolExecutor() as executor:
        executor.submit(add_memory_in_mem0, user_query, user_id)
    
    return assistant_answer, messages


def rephrase_user_query(query, memories) -> str:
    """
    Rephrase the user query to make it more specific and relevant.
    """

    llm = GroqInference()

    memories = "\n".join(memories)

    system_prompt = """
    You are an expert in rephrasing user queries. You are given a user query and some relevant memories.
    You need to rephrase the user query to make it more specific and relevant according to the memory.
    
    Instructions:
    1. Don't assume any information. Just rephrase the user query to make it more specific and relevant according to the memory..
    2. Give the response in the provided JSON FORMAT
    
    JSON FORMAT:
    ```json
    {
        "rephrased_query": "Rephrased user query"
    }
    ```
    """

    user_prompt = f"""
    User Query: {query}
    
    RELEVANT MEMORIES:
    {memories}
    
    Note: Only give the JSON as the response.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = llm.generate_response(messages)
    pre_processed_response = pre_process_the_json_response(response)
    response_object = load_object_from_string(pre_processed_response)

    if response_object is None:
        raise Exception("Failed to extract the relevant memories from the user query.")

    return response_object["rephrased_query"]
