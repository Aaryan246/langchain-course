from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()  # Load environment variables from .env file

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_core.messages import HumanMessage, AIMessage
from langchain_tavily import TavilySearch

llm = ChatOllama(model="gpt-oss:20b", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    res = agent.invoke({"messages": [HumanMessage(content="What is the weather at France?")]})
    print(res)

if __name__ == "__main__":
    main()