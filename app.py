import streamlit as st
import time
# from indexer.indexer import process_file
from retriever import retrieve, format_context, build_rag_chain

# Page configuration
st.set_page_config(page_title="Legal RAG Assistant", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm_model_name" not in st.session_state:
    st.session_state.llm_model_name = "phi3"  # Default LLM model     

# Sidebar menu
page = st.sidebar.selectbox("Choose the page", ["Question Answering", "File Upload"])

if page == "Question Answering":   
    st.header("RAG Question Answering")
    # LLM options: phi3, qwen, deepseek-r1:1.5b
    # Current best LLM: phi3        
    llm_model_name = st.selectbox("Choose LLM model", 
                                  options=["phi3", "qwen", "deepseek-r1:1.5b"], 
                                  index=["phi3", "qwen", "deepseek-r1:1.5b"].index(st.session_state.llm_model_name),
                                  key="llm_model_selector")
    st.session_state.llm_model_name = llm_model_name  # Update session state with the selected model
    st.caption(f"Current LLM model: {llm_model_name}")

    # Show history of interactions
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Enter your question")

    if query:
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""

                # start_retrieval = time.time()
                with st.spinner("Retrieving documents..."):
                    docs = retrieve(query)
                # retrieval_time = time.time() - start_retrieval

                context = format_context(docs)   

                # start_generation = time.time()     
                rag_chain = build_rag_chain(llm_model_name)
                placeholder = st.empty()  # Placeholder to update the answer in real-time
                full_response = ""
                
                with st.spinner(f"Generating answer with {llm_model_name}"):
                    for chunk in rag_chain.stream({"query": query, "context": context}):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")  # Show the answer with a cursor
                    placeholder.markdown(full_response)  # Final answer without cursor
                    st.session_state.message.append({"role": "assistant", "content": full_response})  # Save the assistant's response in session state
                    # answer = rag_chain.invoke({"query": query, "context": context})
                    # generation_time = time.time() - start_generation
                    # st.write(answer)

                # st.markdown("### Performance")
                # st.write(f"Retrieval time: {retrieval_time:.2f} seconds")
                # st.write(f"Generation time: {generation_time:.2f} seconds")

                # with st.expander("Reference Sources"):
                #     for ref in refs:
                #         st.markdown(f"- {ref}")

    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()  

elif page == "File Upload":
    st.header("File Upload & Indexing")
    uploaded_file = st.file_uploader("Please select a file", type=["pdf", "docx", "txt"])

    # if uploaded_file is not None:
    #     with st.spinner("Processing file..."):
    #         # Call the process_file function from the indexer
    #         result = process_file(uploaded_file)
    #     st.success(f"File processed and indexed: {uploaded_file.name}")
    #     st.json(result)  # Can show the chunk result or metadata                