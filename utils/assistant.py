from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Any

class Assistant:
    """
    Wrapper for LLM chains that ensures non-empty responses.
    Retries if the LLM returns an empty response.
    """
    def __init__(self, runnable: Runnable, max_retries: int = 3):
        self.runnable = runnable
        self.max_retries = max_retries

    def invoke(self, messages: List[Any]) -> AIMessage:
        """Invoke the assistant."""
        retry_count = 0
        while retry_count < self.max_retries:
            result = self.runnable.invoke(messages)
            
            # Check if response is empty/invalid
            is_empty = False
            if not result.tool_calls:
                if not result.content:
                    is_empty = True
                elif isinstance(result.content, list):
                    if not result.content or not result.content[0].get("text"):
                        is_empty = True
            
            # If empty, add a retry prompt and try again
            if is_empty:
                retry_count += 1
                if retry_count < self.max_retries:
                    messages = messages + [HumanMessage(content="Respond with a real output.")]
                else:
                    # Last retry - return a default response
                    return AIMessage(content="I apologize, but I'm having trouble generating a response. Please try rephrasing your question.")
            else:
                # Valid response - return it
                return result
        
        # Should never reach here, but just in case
        return AIMessage(content="I apologize, but I'm having trouble generating a response. Please try again.")