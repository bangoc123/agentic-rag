import os
from dotenv import load_dotenv
load_dotenv()


from agents import Agent, Runner, handoff, trace
from agents.extensions.models.litellm_model import LitellmModel
from agents.handoffs import HandoffInputData
from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio

from prompt import MANAGER_INSTRUCTION, PRODUCT_INSTRUCTION, SHOP_INFORMATION_INSTRUCTION
from rag import rag, shop_information_rag



app = Flask(__name__)
CORS(app)

def custom_input_filter(input_data: HandoffInputData) -> HandoffInputData:
        # modified_data = HandoffInputData(
        #     input_history=input_data.input_history,  # Keep history as it is
        #     pre_handoff_items=(),
        #     new_items=()
        # )
    return input_data


api_key = os.getenv("TOGETHER_API_KEY")

# Tạo agents với LiteLLM và Kimi model
print("Creating agents with LiteLLM and Kimi-K2-Instruct...")
try:
    
    product_agent = Agent(
        name="product",
        instructions=PRODUCT_INSTRUCTION,
        tools=[rag],
        model=LitellmModel(
                model="together_ai/OpenAI/gpt-oss-20B",
                api_key=api_key
            )
    )

    shop_information_agent = Agent(
        name="shop_information",
        instructions=SHOP_INFORMATION_INSTRUCTION,
        tools=[shop_information_rag],
        model=LitellmModel(
                model="together_ai/OpenAI/gpt-oss-20B",
                api_key=api_key
            )
    )

    manager_agent = Agent(
        name="manager",
        instructions=MANAGER_INSTRUCTION,
        handoffs=[
            handoff(
                product_agent,
                input_filter=custom_input_filter,
            ),
            shop_information_agent
        ],
        model=LitellmModel(
                model="together_ai/moonshotai/Kimi-K2-Instruct",
                api_key=api_key
            )
    )
    print("✓ Agents created successfully with LiteLLM and Kimi-K2-Instruct")

except Exception as e:
    print(f"✗ Error creating agents: {e}")
    exit(1)

conversation_history = {}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("message", "")
    thread_id = data.get("thread_id", "1")
    
    if not query:
        return jsonify({"error": "Missing query parameter"}), 400
    
    if thread_id not in conversation_history:
        conversation_history[thread_id] = []
    
    try:
        with trace(workflow_name="Conversation", group_id=thread_id):
            new_input = conversation_history[thread_id] + [{"role": "user", "content": query}]
            result = asyncio.run(Runner.run(manager_agent, new_input))
            conversation_history[thread_id] = new_input + [{"role": "assistant", "content": str(result.final_output)}]
        
        return jsonify({
            "role": "assistant",
            "content": str(result.final_output)
        })
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            "error": f"Error processing request: {str(e)}"
        }), 500



if __name__ == "__main__":
    # Kiểm tra API key
    if not os.getenv("TOGETHER_API_KEY"):
        print("Warning: TOGETHER_API_KEY not found in environment variables")
        exit(1)
    else:
        print(f"✓ TOGETHER_API_KEY loaded: {os.getenv('TOGETHER_API_KEY')[:8]}...")
    
    print(f"✓ Using LiteLLM with Kimi-K2-Instruct model")
    
    app.run(host="0.0.0.0", port=5001, debug=True)