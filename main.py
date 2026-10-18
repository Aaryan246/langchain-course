from operator import itemgetter
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama

load_dotenv()

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatOllama(model="gemma4:e4b", temperature=0)
embeddings = OllamaEmbeddings(
        model="qwen3-embedding:4b",
        dimensions=1536
    )

vector_store = PineconeVectorStore(
        index_name= os.environ.get("INDEX_NAME"),
        embedding=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based on the context below.

Context: {context}

Question: {query}

Provide a detailed answer to the question."""
)

def retrieval_chain_without_lcel(query: str):
    """
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """

    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: format documents into context string
    context = format_docs(docs)

    # Step 3: format the prompt with context and question
    messages = prompt_template.format_messages(context=context, query=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content

def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}
    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.airnvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context = itemgetter("query") | retriever | format_docs
        ) |
        prompt_template |
        llm | StrOutputParser()
    )
    return retrieval_chain

if __name__ == "__main__":
    query = "What is Pinecone in machine learning"

    print("Result without LCEL")
    res_without_lecl = retrieval_chain_without_lcel(query)
    print(res_without_lecl)

    print("Result with LCEL")
    chain_with_lcel = create_retrieval_chain_with_lcel()
    res = chain_with_lcel.invoke({"query":query})
    print(res)