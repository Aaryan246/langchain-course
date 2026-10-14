from dotenv import load_dotenv
import ollama

load_dotenv()

from langchain.chat_models import init_chat_model
from langsmith import traceable

MAX_ITERATIONS = 3
MODEL = "qwen2.5:7b"


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"Looking up price for product: {product}")
    prices = {
        "apple": 100.0,
        "banana": 50.0,
        "orange": 75.0,
    }
    return prices.get(product.lower(), 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price based on the discount tier
    Available discount tiers: silver, gold, platinum"""
    print(f"Applying discount: {discount_tier} to price: {price}")
    discount_percentages = {
        "silver": 0.05,
        "gold": 0.10,
        "platinum": 0.15,
    }
    discount = discount_percentages.get(discount_tier.lower(), 0.0)
    discounted_price = price * (1 - discount)
    return round(discounted_price, 2)

# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    }
]

# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.


@traceable(name="Ollama chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(messages=messages, tools=tools_for_llm, model=MODEL)


@traceable(name="Ollama agent loop")
def run_agent(query: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using maths."
                "Always use the apply_discount tool to calculate the discounted price.\n"
                "4. If user does not specify a discount tier, ask them which tier - do not assume yourself.\n"
            ),
        },
        {"role": "user", "content": query},
    ]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\nIteration {i} of {MAX_ITERATIONS}")

        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages)
        ai_message = response.message
        
        tool_calls = ai_message.tool_calls
        
        # If not tool calls, we have a final answer
        if not tool_calls:
            print(f"Final answer from agent: {ai_message.content}")
            return ai_message.content

        # Process tool calls
        tool_call = tool_calls[0]
        # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"Tool selected {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        # Difference 7: Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)

        print(f"Tool Result : {observation}")

        messages.append(ai_message)
        messages.append({
            "role": "tool",
            "content" : str(observation)
        })

    print("Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("Starting agent...")
    query = "What is the price of an apple with a gold discount?"
    run_agent(query)
