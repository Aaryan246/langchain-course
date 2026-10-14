from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 3
MODEL = 'qwen2.5:7b'

@tool
def get_product_price(product:str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"Looking up price for product: {product}")
    prices = {
        "apple": 100.0,
        "banana": 50.0,
        "orange": 75.0,
    }
    return prices.get(product.lower(),0)

@tool 
def apply_discount(price:float, discount_tier:str)-> float:
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

@traceable(name="Lanchain agent loop")
def run_agent(query:str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
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
        ),
        HumanMessage(content=query)
    ]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\nIteration {i} of {MAX_ITERATIONS}")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # If not tool calls, we have a final answer
        if not tool_calls:
            print(f"Final answer from agent: {ai_message.content}")
            return ai_message.content


        # Process tool calls
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool selected {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"Tool Result : {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call_id
        ))

    print("Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("Starting agent...")
    query = "What is the price of an apple with a gold discount?"
    run_agent(query)