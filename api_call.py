import requests
import json
import uuid
from typing import Optional, Dict, Any
import time

class ChatAPIClient:
    def __init__(self, base_url: str = "http://localhost:5001"):
        """
        Initialize the chat API client
        
        Args:
            base_url: The base URL of the Flask API server
        """
        self.base_url = base_url.rstrip('/')
        self.chat_endpoint = f"{self.base_url}/chat"
        self.session = requests.Session()
        
    def send_message(self, message: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a message to the chat API
        
        Args:
            message: The message to send
            thread_id: Optional thread ID for conversation continuity
            
        Returns:
            Dictionary containing the API response
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())
            
        payload = {
            "message": message,
            "thread_id": thread_id
        }
        
        try:
            response = self.session.post(
                self.chat_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60  # 60 second timeout
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "thread_id": thread_id,
                    "response": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "thread_id": thread_id
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out",
                "thread_id": thread_id
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": f"Could not connect to server at {self.base_url}",
                "thread_id": thread_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "thread_id": thread_id
            }
    
    def test_connection(self) -> bool:
        """
        Test if the API server is reachable
        
        Returns:
            True if server is reachable, False otherwise
        """
        try:
            # Try to connect to the base URL
            response = self.session.get(f"{self.base_url}/", timeout=5)
            return True
        except:
            return False

class ConversationManager:
    def __init__(self, client: ChatAPIClient):
        """
        Manage conversations with the chat API
        
        Args:
            client: ChatAPIClient instance
        """
        self.client = client
        self.current_thread_id = None
        self.conversation_history = []
    
    def start_new_conversation(self) -> str:
        """
        Start a new conversation thread
        
        Returns:
            New thread ID
        """
        self.current_thread_id = str(uuid.uuid4())
        self.conversation_history = []
        return self.current_thread_id
    
    def send_message(self, message: str) -> Dict[str, Any]:
        """
        Send a message in the current conversation
        
        Args:
            message: Message to send
            
        Returns:
            API response
        """
        if self.current_thread_id is None:
            self.start_new_conversation()
        
        result = self.client.send_message(message, self.current_thread_id)
        
        if result["success"]:
            # Store conversation history locally
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({
                "role": "assistant", 
                "content": result["response"].get("content", "")
            })
        
        return result
    
    def get_conversation_history(self) -> list:
        """
        Get the local conversation history
        
        Returns:
            List of conversation messages
        """
        return self.conversation_history.copy()
    
    def print_conversation(self):
        """
        Print the conversation history in a readable format
        """
        print("\n" + "="*50)
        print("CONVERSATION HISTORY")
        print("="*50)
        
        for i, msg in enumerate(self.conversation_history):
            role = msg["role"].upper()
            content = msg["content"]
            print(f"\n[{role}]:")
            print(content)
            
            if i < len(self.conversation_history) - 1:
                print("-" * 30)
        
        print("="*50 + "\n")

def interactive_chat():
    """
    Interactive command-line chat interface
    """
    print("🤖 Chat API Client")
    print("Type 'quit' to exit, 'new' for new conversation, 'history' to see conversation")
    print("-" * 50)
    
    # Initialize client and conversation manager
    client = ChatAPIClient()
    conversation = ConversationManager(client)
    
    # Test connection
    if not client.test_connection():
        print("❌ Could not connect to API server. Make sure it's running on http://localhost:5001")
        return
    else:
        print("✅ Connected to API server")
    
    # Start first conversation
    thread_id = conversation.start_new_conversation()
    print(f"Started new conversation: {thread_id[:8]}...")
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'new':
                thread_id = conversation.start_new_conversation()
                print(f"🆕 Started new conversation: {thread_id[:8]}...")
                continue
            elif user_input.lower() == 'history':
                conversation.print_conversation()
                continue
            elif not user_input:
                continue
            
            print("⏳ Thinking...")
            start_time = time.time()
            
            result = conversation.send_message(user_input)
            
            elapsed_time = time.time() - start_time
            
            if result["success"]:
                response_content = result["response"].get("content", "No response content")
                print(f"🤖 Assistant: {response_content}")
                print(f"⏱️  Response time: {elapsed_time:.2f}s")
            else:
                print(f"❌ Error: {result['error']}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def example_usage():
    """
    Example of how to use the client programmatically
    """
    print("📝 Example API Usage")
    print("-" * 30)
    
    # Initialize client
    client = ChatAPIClient()
    
    # Test connection
    if not client.test_connection():
        print("❌ API server not available")
        return
    
    # Create conversation manager
    conversation = ConversationManager(client)
    
    # Example conversation
    messages = [
        "Hello! Can you help me with product information?",
        "What products do you have available?",
        "Tell me about your shop information."
    ]
    
    for msg in messages:
        print(f"\n💬 Sending: {msg}")
        result = conversation.send_message(msg)
        
        if result["success"]:
            response = result["response"].get("content", "No response")
            print(f"🤖 Response: {response}")
        else:
            print(f"❌ Error: {result['error']}")
            break
    
    print(f"\n📊 Conversation ID: {conversation.current_thread_id}")
    print(f"📝 Total messages: {len(conversation.conversation_history)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "example":
            example_usage()
        elif sys.argv[1] == "test":
            client = ChatAPIClient()
            if client.test_connection():
                print("✅ API server is reachable")
            else:
                print("❌ API server is not reachable")
        else:
            print("Usage: python client.py [example|test]")
    else:
        interactive_chat()