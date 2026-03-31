import streamlit as st
# import time
# from indexer.indexer import process_file

try:
    from contract_copilot.config import config
    from contract_copilot.model import build_rag_chain
    from contract_copilot.retriever import format_context, retrieve
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent
    SRC_ROOT = PROJECT_ROOT / "src"
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from contract_copilot.config import config
    from contract_copilot.model import build_rag_chain
    from contract_copilot.retriever import format_context, retrieve

# Page configuration
st.set_page_config(page_title="Legal RAG Assistant", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm_model_name" not in st.session_state:
    st.session_state.llm_model_name = config.default_llm_model_name


def get_doc_title(doc):
    return doc.metadata.get("title") or doc.metadata.get("document_title") or "Unknown Title"


def get_doc_source(doc):
    return doc.metadata.get("source") or doc.metadata.get("source_path") or "Unknown Source"

# Sidebar menu
page = st.sidebar.selectbox("Choose the page", ["Question Answering", "File Upload"])

if page == "Question Answering":   
    st.header("RAG Question Answering")
    # LLM options: phi3, qwen, deepseek-r1:1.5b
    # Current best LLM: phi3        
    llm_options = [config.default_llm_model_name, "qwen", "deepseek-r1:1.5b"]
    llm_model_name = st.selectbox(
        "Choose LLM model",
        options=llm_options,
        index=llm_options.index(st.session_state.llm_model_name),
        key="llm_model_selector",
    )
    st.session_state.llm_model_name = llm_model_name  # Update session state with the selected model
    st.caption(f"Current LLM model: {llm_model_name}")

    # Show history of interactions
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("Reference Sources", expanded=False):
                    for i, doc in enumerate(message["sources"], start=1):
                        st.markdown(f"**[{i}] {get_doc_title(doc)}, ID: {doc.id}**")
                        st.caption(get_doc_source(doc))
                        st.write(doc.page_content)

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

                if not docs:
                    placeholder.markdown("No relevant documents were retrieved from Qdrant.")
                    st.session_state.messages.append({"role": "assistant", "content": "No relevant documents were retrieved from Qdrant.", "sources": []})
                    st.stop()
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
                    # answer = rag_chain.invoke({"query": query, "context": context})
                    # generation_time = time.time() - start_generation
                    # st.write(answer)

                with st.expander("Reference Sources", expanded=False):
                    for i, doc in enumerate(docs, start=1):
                        st.markdown(f"**[{i}] {get_doc_title(doc)}, ID: {doc.id}**")
                        st.caption(get_doc_source(doc))
                        st.write(doc.page_content)

                st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": docs})  # Save the assistant's response in session state

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
