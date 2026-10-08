from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()  # Load environment variables from .env file

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_core.messages import HumanMessage, AIMessage
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )
llm = ChatOllama(model="gemma4:e4b", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    res = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer in Mumbai on linkedin and list their details?")]})
    print(res)

if __name__ == "__main__":
    main()