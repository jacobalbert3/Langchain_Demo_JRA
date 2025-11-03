"""
Test script for Assistant wrapper to verify it handles empty responses correctly.
"""
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from utils.assistant import Assistant

def test_assistant_normal_operation():
    """Test that Assistant works normally when LLM returns valid responses."""
    print("=" * 60)
    print("Test 1: Normal Operation (Valid Response)")
    print("=" * 60)
    
    class ValidResponseRunnable(Runnable):
        """Mock runnable that always returns valid responses."""
        def invoke(self, messages, config=None):
            return AIMessage(content="This is a valid response.")
    
    mock_runnable = ValidResponseRunnable()
    assistant = Assistant(mock_runnable)
    messages = [HumanMessage(content="Hello, can you help me?")]
    
    result = assistant.invoke(messages)
    
    print(f"✓ Received response: {type(result).__name__}")
    print(f"✓ Has content: {bool(result.content)}")
    print(f"✓ Content: {result.content}")
    
    if result.content:
        print("\n✅ Test 1 PASSED: Assistant works with valid responses\n")
        return True
    else:
        print("\n❌ Test 1 FAILED: Assistant did not return valid response\n")
        return False


def test_assistant_with_empty_response_mock():
    """Test that Assistant retries when LLM returns empty responses."""
    print("=" * 60)
    print("Test 2: Retry Logic (Empty Response Handling)")
    print("=" * 60)
    
    class EmptyResponseRunnable(Runnable):
        """Mock runnable that returns empty responses first, then valid."""
        def __init__(self):
            self.call_count = 0
        
        def invoke(self, messages, config=None):
            self.call_count += 1
            if self.call_count == 1:
                # First call: empty response
                print(f"  Call {self.call_count}: Returning empty response")
                return AIMessage(content="")
            elif self.call_count == 2:
                # Second call: empty list response
                print(f"  Call {self.call_count}: Returning empty list response")
                return AIMessage(content=[])
            else:
                # Third call: valid response
                print(f"  Call {self.call_count}: Returning valid response")
                return AIMessage(content="This is a valid response after retries.")
    
    mock_runnable = EmptyResponseRunnable()
    assistant = Assistant(mock_runnable, max_retries=3)
    messages = [HumanMessage(content="Test message")]
    
    result = assistant.invoke(messages)
    
    print(f"\n✓ Total calls made: {mock_runnable.call_count}")
    print(f"✓ Final response content: {result.content}")
    print(f"✓ Has valid content: {bool(result.content)}")
    
    if mock_runnable.call_count == 3 and result.content:
        print("\n✅ Test 2 PASSED: Assistant correctly retries on empty responses\n")
        return True
    else:
        print("\n❌ Test 2 FAILED: Assistant did not retry correctly\n")
        return False


def test_assistant_max_retries():
    """Test that Assistant returns fallback after max retries."""
    print("=" * 60)
    print("Test 3: Max Retries (Fallback Response)")
    print("=" * 60)
    
    class AlwaysEmptyRunnable(Runnable):
        """Mock runnable that always returns empty responses."""
        def __init__(self):
            self.call_count = 0
        
        def invoke(self, messages, config=None):
            self.call_count += 1
            print(f"  Call {self.call_count}: Returning empty response")
            return AIMessage(content="")
    
    mock_runnable = AlwaysEmptyRunnable()
    assistant = Assistant(mock_runnable, max_retries=3)
    messages = [HumanMessage(content="Test message")]
    
    result = assistant.invoke(messages)
    
    print(f"\n✓ Total calls made: {mock_runnable.call_count}")
    print(f"✓ Final response: {result.content}")
    print(f"✓ Is fallback message: {'trouble generating' in str(result.content).lower()}")
    
    if mock_runnable.call_count == 3 and "trouble generating" in str(result.content).lower():
        print("\n✅ Test 3 PASSED: Assistant returns fallback after max retries\n")
        return True
    else:
        print("\n❌ Test 3 FAILED: Assistant did not handle max retries correctly\n")
        return False


def test_assistant_with_tool_calls():
    """Test that Assistant accepts responses with tool_calls (not empty)."""
    print("=" * 60)
    print("Test 4: Tool Calls (Valid Response)")
    print("=" * 60)
    
    class ToolCallRunnable(Runnable):
        """Mock runnable that returns tool calls."""
        def invoke(self, messages, config=None):
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": "test_tool", "args": {"param": "value"}}]
            return msg
    
    mock_runnable = ToolCallRunnable()
    assistant = Assistant(mock_runnable)
    messages = [HumanMessage(content="Test message")]
    
    result = assistant.invoke(messages)
    
    print(f"✓ Response type: {type(result).__name__}")
    print(f"✓ Has tool_calls: {bool(hasattr(result, 'tool_calls') and result.tool_calls)}")
    
    if hasattr(result, 'tool_calls') and result.tool_calls:
        print(f"✓ Tool calls found: {len(result.tool_calls)}")
        print("\n✅ Test 4 PASSED: Assistant accepts responses with tool_calls\n")
        return True
    else:
        print("\n❌ Test 4 FAILED: Assistant did not handle tool_calls correctly\n")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ASSISTANT WRAPPER TEST SUITE")
    print("=" * 60 + "\n")
    
    results = []
    
    try:
        results.append(("Normal Operation", test_assistant_normal_operation()))
    except Exception as e:
        print(f"❌ Test 1 FAILED with error: {e}\n")
        results.append(("Normal Operation", False))
    
    try:
        results.append(("Retry Logic", test_assistant_with_empty_response_mock()))
    except Exception as e:
        print(f"❌ Test 2 FAILED with error: {e}\n")
        results.append(("Retry Logic", False))
    
    try:
        results.append(("Max Retries", test_assistant_max_retries()))
    except Exception as e:
        print(f"❌ Test 3 FAILED with error: {e}\n")
        results.append(("Max Retries", False))
    
    try:
        results.append(("Tool Calls", test_assistant_with_tool_calls()))
    except Exception as e:
        print(f"❌ Test 4 FAILED with error: {e}\n")
        results.append(("Tool Calls", False))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 60 + "\n")

