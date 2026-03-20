import streamlit as st
import time
# from indexer.indexer import process_file
from retriever import retrieve, format_context, build_rag_chain

# Page configuration
st.set_page_config(page_title="RAG Demo", layout="wide")

# Sidebar menu
page = st.sidebar.selectbox("Choose the page", ["File Upload", "Question Answering"])

if page == "Question Answering":
    st.header("RAG Question Answering")
    # LLM options: phi3, qwen, deepseek-r1:1.5b
    # Current best LLM: phi3        
    llm_model_name = st.selectbox("Choose LLM model", options=["phi3", "qwen", "deepseek-r1:1.5b"], index=0, key="llm_model_selector")
    st.caption(f"Current LLM model: {llm_model_name}")
    query = st.text_input("Enter your question")

    if st.button("Submit Query"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            st.subheader("Answer")
            placeholder = st.empty()
            full_response = ""

            start_retrieval = time.time()
            with st.spinner("Retrieving documents..."):
                docs = retrieve(query)
            retrieval_time = time.time() - start_retrieval

            context = format_context(docs)   

            start_generation = time.time()     
            rag_chain = build_rag_chain(llm_model_name)
            
            with st.spinner(f"Generating answer with {llm_model_name}"):
                for chunk in rag_chain.stream({"query": query, "context": context}):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")  # Show the answer with a cursor
                generation_time = time.time() - start_generation
                placeholder.markdown(full_response)  # Final answer without cursor

            st.markdown("### Performance")
            st.write(f"Retrieval time: {retrieval_time:.2f} seconds")
            st.write(f"Generation time: {generation_time:.2f} seconds")
            # with st.expander("Reference Sources"):
            #     for ref in refs:
            #         st.markdown(f"- {ref}")

elif page == "File Upload":
    st.header("File Upload & Indexing")
    uploaded_file = st.file_uploader("Please select a file", type=["pdf", "docx", "txt"])

    # if uploaded_file is not None:
    #     with st.spinner("Processing file..."):
    #         # Call the process_file function from the indexer
    #         result = process_file(uploaded_file)
    #     st.success(f"File processed and indexed: {uploaded_file.name}")
    #     st.json(result)  # Can show the chunk result or metadata                