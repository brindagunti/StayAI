# from backend.conversation.chat import chat_with_travel_assistant

# if __name__ == "__main__":
#     try:
#         chat_with_travel_assistant()
#     except Exception as e:
#         print(f"Error in chat: {str(e)}")

from backend.conversation.chat import chat_with_travel_assistant
from backend.memory.chroma_memory.add_data import add_pdf_to_chroma
from backend.memory.mem0_memory.try_mem0 import add_memory_in_mem0, extract_relevant_memories
from backend.agents.stay_ai_crew.simple_agent_framework.browser_agent import BrowserTool
from backend.agents.stay_ai_crew.simple_agent_framework.browser_agent import BrowserAgent

if __name__ == "__main__":
    
    agent = BrowserAgent()
    agent.run("Give me an iteinary for a trip to Hyderabad")