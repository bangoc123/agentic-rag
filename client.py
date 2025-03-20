import streamlit as st
import requests
import json
import uuid
from typing import List, Dict, Any

st.logo("https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/678dadd0-603b-11ef-b0a7-998b84b38d43-ProtonX_logo_horizontally__1_.png")

# Configure page settings
st.set_page_config(
    page_title="Chat Application",
    page_icon="💬",
    layout="centered"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.assistant {
        background-color: #475063;
        color: white;
    }
    .chat-message .message-content {
        display: flex;
        margin-top: 0;
    }
    .message-content p {
        margin-bottom: 0;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Set API endpoint
API_ENDPOINT = "http://localhost:5001/chat"

# Add headers for CORS if needed
HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*"
}

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Title for the app
st.title("💬 Agentic RAG Application")

# Function to display chat messages
def display_messages():
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        avatar_url = "https://api.dicebear.com/7.x/bottts/svg?seed=assistant" if role == "assistant" else "https://api.dicebear.com/7.x/personas/svg?seed=user"
        
        with st.container():
            st.markdown(f"""
            <div class="chat-message {role}">
                <div class="message-content">
                    <img class="avatar" src="{avatar_url}">
                    <div>{content}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Display existing chat messages
display_messages()

# Function to send message and get response
def send_message():
    if st.session_state.user_input:
        user_message = st.session_state.user_input
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        # Clear input field
        st.session_state.user_input = ""
        
        # Prepare data for API call
        data = {
            "message": user_message,
            "thread_id": st.session_state.thread_id
        }
        
        try:
            # Show spinner while waiting for response
            # with st.spinner("Thinking..."):
            response = requests.post(API_ENDPOINT, json=data, headers=HEADERS)
            
            if response.status_code == 200:
                assistant_response = response.json()
                # Add assistant response to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response["content"]
                })
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to communicate with the server: {str(e)}")

# Create a form for user input
with st.form(key="chat_form", clear_on_submit=True):
    st.text_input(
        "Your message:",
        key="user_input",
        placeholder="Type your message here...",
    )
    submit_button = st.form_submit_button("Send", on_click=send_message)

# Add a sidebar with options
with st.sidebar:
    st.title("Options")
    
    if st.button("New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.experimental_rerun()
    
    st.markdown("---")
    st.write("Current Thread ID:", st.session_state.thread_id)