from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
import os

load_dotenv()


if __name__ == "__main__":
    loader = UnstructuredLoader("mediumblog1.txt", chunking_strategy='basic', max_characters=100000)
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)

    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:4b",
        dimensions=1536
    )

    index_name = os.environ.get("INDEX_NAME")

    print("Generating local Nomic embeddings and uploading to Pinecone...")
    vector_store = PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=index_name
    )

    print("Done")