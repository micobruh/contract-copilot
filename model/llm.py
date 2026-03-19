from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = OllamaLLM(
    model="llama3"
)

prompt = ChatPromptTemplate.from_template(
"""
You are a helpful AI assistant.

Use the following retrieved context to answer the question.

Context:
{context}

Question:
{question}

Answer concisely in no more than three sentences.
"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)