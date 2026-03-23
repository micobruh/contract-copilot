import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import OllamaLLM
import streamlit as st
from ..config import config


@st.cache_resource
def load_llm(llm_model_name=config.default_llm_model_name):
    return OllamaLLM(
        model=llm_model_name,
        streaming=True,
        base_url=os.getenv("OLLAMA_HOST"),
    )


@st.cache_resource
def build_rag_chain(llm_model_name=config.default_llm_model_name):
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful Law AI assistant.

        Given this question: {query}

        Answer using ONLY the provided document sources: {context} 

        Extract the most relevant passage from the retrieved documents that answers the question.

        If the answer isn't there, say "I don't know." Do not use prior knowledge.
        """
    )
    llm = load_llm(llm_model_name)
    return (
        prompt
        | llm
        | StrOutputParser()
    )


# @st.cache_resource
# def build_rag_chain(llm_model_name):
#     prompt = ChatPromptTemplate.from_template(
#         """
#         You are a helpful Law AI assistant.

#         Given this question: {query}

#         Answer using ONLY the provided document sources: {context} 

#         Extract the most relevant passage from the retrieved documents that answers the question.

#         If the answer isn't there, say "I don't know." Do not use prior knowledge.
#         """
#     )
#     retriever = RunnableLambda(retrieve)
#     llm = load_llm(llm_model_name)
#     rag_chain = (
#         {
#             "context": retriever | format_context,
#             "query": RunnablePassthrough(),
#         }
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     return rag_chain


def answer_question(query, llm_model_name=config.default_llm_model_name):
    rag_chain = build_rag_chain(llm_model_name)
    return rag_chain.invoke(query)
