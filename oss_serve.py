import os
from dotenv import load_dotenv
load_dotenv()


from agents import Agent, Runner, handoff, trace, ModelSettings
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

def require_env_key(key_name: str) -> str:
    value = os.getenv(key_name)
    if not value:
        print(f"Warning: {key_name} not found in environment variables")
        raise EnvironmentError(f"{key_name} not found in environment variables")
    print(f"✓ {key_name} loaded: {value[:8]}...")
    return value


togetherai_key = require_env_key("TOGETHER_API_KEY")
groq_key = require_env_key("GROQ_API_KEY")

# Tạo agents với LiteLLM và Kimi model
print("Creating agents with LiteLLM...")
try:
    kimi = LitellmModel(
                model="together_ai/moonshotai/Kimi-K2-Instruct",
                api_key=togetherai_key
            )
    gpt_oss_20b = LitellmModel(
                model="groq/openai/gpt-oss-20b",
                api_key=groq_key
            )
    gpt_oss_120b = LitellmModel(
                model="groq/openai/gpt-oss-120b",
                api_key=groq_key
            )
    product_agent = Agent(
        name="product",
        instructions=PRODUCT_INSTRUCTION,
        tools=[rag],
        model=gpt_oss_20b,
        model_settings=ModelSettings(tool_choice="required")
    )

    shop_information_agent = Agent(
        name="shop_information",
        instructions=SHOP_INFORMATION_INSTRUCTION,
        tools=[shop_information_rag],
        model=gpt_oss_20b,
        model_settings=ModelSettings(tool_choice="required")
    )

    manager_agent = Agent(
        name="manager",
        instructions=MANAGER_INSTRUCTION,
        handoffs=[
            product_agent,
            shop_information_agent
        ],
        model=kimi
    )
    print("✓ Agents created successfully with LiteLLM")

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
            # print(f"đoạn thoại: {new_input}")
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
    app.run(host="0.0.0.0", port=5001, debug=True)