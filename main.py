from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

def main():
    load_dotenv()  # Load environment variables from .env file

    summary_template = """given information: {info} about a person, I want you to
    1. A short summary of the person in 1-2 sentences.
    2. two interesting facts about the person."""

    # Define a prompt template
    prompt_template = PromptTemplate(
        input_variables=["info"],
        template=summary_template
    )

    information = """
Elon Reeve Musk (born June 28, 1971) is a businessman, industrialist, and former public official who is the chief executive officer (CEO) and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and briefly became the only trillionaire (in terms of US dollars) in June 2026; as of October 2026, Forbes estimates his net worth to be US$1 trillion.

Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002."""

    llm = ChatOllama(model="gpt-oss:20b", temperature=0)
    chain = prompt_template | llm
    res = chain.invoke(input={"info": information})
    print(res.content)


if __name__ == "__main__":
    main()