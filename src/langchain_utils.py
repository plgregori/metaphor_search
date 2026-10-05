from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.sklearn import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings

'''This file contains helper functions that are used in the 'true' RAG pipeline.
They are inspired by the old RAG notebooks. Everything in this file should probably be moved to rag_utils.py.'''

def setup_rag_ollama(rag_context: str) -> SKLearnVectorStore:
        # Create documents from context
        docs = [Document(page_content=rag_context)]
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(docs)
        
        # Create vector store
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=OllamaEmbeddings(model='nomic-embed-text'),
        )
        
        return vectorstore

def retrieve_codebook_context(vectorstore: SKLearnVectorStore, query_text: str, top_k: int = 1) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(query_text)
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def build_codebook_rag_messages(context: str, query_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system",
         "content": ("You are a helpful AI assistant. Use the following pieces of context to answer "
                     "the question at the end. If you don't know the answer, just say you don't know. "
                     "DO NOT try to make up an answer. If the question is not related to the context, "
                     "politely respond that you are tuned to only answer questions that are related to "
                     "the context.\n" + context)},
        {"role": "system",
         "content": ("You are a linguistic expert trained in metaphor identification. When the user "
                     "provides a text, follow this protocol:\n"
                     "\u2022 Identify all metaphorical expressions.\n"
                     "\u2022 Wrap each one in <Metaphor> and </Metaphor> tags.\n"
                     "\u2022 Reproduce the rest of the text exactly as written.\n"
                     "\u2022 Do not include any explanation, commentary, or extra content in this message.")},
        {"role": "user",
         "content": f"Can you please identify and tag the metaphors in the following text?\n{query_text}"},
    ]