

# Mobile Shop Sales Multi-Agent System

This project utilizes OpenAI Agents SDK to build a multi-agent system for handling inquiries about mobile shop sales. The system consists of multiple specialized agents that work together to provide accurate product information and shop details.

![](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/2cf5e820-ffed-11ef-abed-590e34770dd0-Screen_Shot_2025_03_13_at_11.50.57.png)


## System Architecture

The multi-agent system includes:

1. **Product Agent** - Handles queries related to mobile products, such as specifications and pricing.
2. **Shop Information Agent** - Provides details about shop locations, stock availability, and policies.
3. **Manager Agent** - Orchestrates interactions between agents and delegates requests to the appropriate agent.

### Agent Setup

```python
product_agent = Agent(
    name="product",
    instructions=PRODUCT_INSTRUCTION,
    tools=[rag]
)

shop_information_agent = Agent(
    name="shop_information",
    instructions=SHOP_INFORMATION_INSTRUCTION,
    tools=[get_shop_information]
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
```

## Installation & Setup

### Prerequisites
Ensure you have Python installed and then install the required dependencies:

```sh
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file and add your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

### Setting Up RAG Data
Run the following command to set up data for retrieval-augmented generation (RAG):

```sh
python setup.py
```

### Running the Server
Start the server using:

```sh
python serve.py
```

## API Usage

### Endpoint
**URL:** `http://127.0.0.1:5001/chat`  
**Method:** `POST`

### Request Body
```json
{
    "message": "Điện thoại này bộ nhớ trong bao nhiêu?",
    "thread_id": "1"
}
```

### Testing with cURL
Run the following command to test the server:

```sh
curl -X POST "http://127.0.0.1:5001/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Điện thoại này bộ nhớ trong bao nhiêu?", "thread_id": "1"}'
```

## License
This project is licensed under [MIT License](LICENSE).

---

With this README, your project is now more structured and easier to understand for new users!

