# CHANGE 1: Add re + inspect — we'll parse tool calls from raw text instead of structured JSON.
import re
import inspect
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
    price = float(price)
    discount_percentages = {
        "silver": 0.05,
        "gold": 0.10,
        "platinum": 0.15,
    }
    discount = discount_percentages.get(discount_tier.lower(), 0.0)
    discounted_price = price * (1 - discount)
    return round(discounted_price, 2)

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

# CHANGE 3: Delete the JSON schemas. Tools now live inside the prompt as plain text.
# We derive descriptions from the functions themselves using inspect.

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # wrapped_ bypasses decorator wrappers (e.g., @traceable adds *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES - you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price - 
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


# CHANGE 4: Drop tools= from ollama.chat(). The LLM has no idea it's an agent —
# all agency comes from the prompt above and our regex parsing below.

@traceable(name="Ollama chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(messages=messages, options=options, model=model)


@traceable(name="Ollama agent loop")
def run_agent(query: str):

    # CHANGE 5: One prompt string replaces the system/user message split.
    prompt = react_prompt.format(question=query)
    scratchpad = ""

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\nIteration {i} of {MAX_ITERATIONS}")

        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role":"user","content":full_prompt}],
            options={"stop":["\nObservation"],"temperature":0}
        )

        output = response.message.content

        print(f"[Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"[Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        # CHANGE 6: Parse tool calls from raw text with regex — fragile if LLM doesn't follow format.  
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))


        print(f"  [Tool Result] {observation}")

        # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"


    print("Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("Starting agent...")
    query = "What is the price of an apple with a gold discount?"
    run_agent(query)
