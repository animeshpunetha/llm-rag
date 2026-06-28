from langchain_core.messages import HumanMessage, AIMessage

class SessionMemory:
    def __init__(self, max_turns: int = 3):
        """
        Initializes the memory buffer. 
        max_turns = 3 means remembering the last 3 user questions and 3 AI answers (6 messages total).
        """

        self.history = []
        self.max_turns = max_turns

    def add_turn(self, user_query:str, ai_response: str):
        """Appends the latest interaction to the history and prunes old ones."""
        self.history.append(HumanMessage(content=user_query))
        self.history.append(AIMessage(content = ai_response))

        # Prune if history exceeds our configured limit (1 turn = 2 messages)
        max_messages = self.max_turns*2

        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    
    def get_history(self):
        """Returns the current rolling window of messages."""
        return self.history