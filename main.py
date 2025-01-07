from backend.conversation.chat import chat_with_travel_assistant

if __name__ == "__main__":
    try:
        chat_with_travel_assistant()
    except Exception as e:
        print(f"Error in chat: {str(e)}")