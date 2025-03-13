from agents import Agent, Runner, function_tool, handoff, RunContextWrapper, trace
from agents.extensions import handoff_filters
from agents.handoffs import HandoffInputData
from flask import Flask, request, jsonify
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from prompt import MANAGER_INSTRUCTION, PRODUCT_INSTRUCTION, SHOP_INFORMATION_INSTRUCTION
from rag import rag

app = Flask(__name__)

def custom_input_filter(input_data: HandoffInputData) -> HandoffInputData:
    # modified_data = HandoffInputData(
    #     input_history=input_data.input_history,  # Keep history as it is
    #     pre_handoff_items=(),
    #     new_items=()
    # )
    return input_data


@function_tool
def get_shop_information(shop_info: str) -> dict:
    return {
        "content": shop_info
    }

product_agent = Agent(
    name="product",
    instructions=PRODUCT_INSTRUCTION,
    tools=[
        rag,
    ]
)

shop_information_agent = Agent(
    name="shop_information",
    instructions=SHOP_INFORMATION_INSTRUCTION,
    tools=[
        get_shop_information
    ]
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
    ]
    
)


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

    with trace(workflow_name="Conversation", group_id=thread_id):
        new_input = conversation_history[thread_id] + [{"role": "user", "content": query}]

        result = asyncio.run(Runner.run(manager_agent, new_input))
        conversation_history[thread_id] = new_input + [{"role": "assistant", "content": str(result.final_output)}]

    return jsonify({
        "role": "assistant",
        "content": str(result.final_output)
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
