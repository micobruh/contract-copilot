import streamlit as st
# from indexer.indexer import process_file
from retriever.retriever import answer_question

# Page configuration
st.set_page_config(page_title="RAG Demo", layout="wide")

# Sidebar menu
page = st.sidebar.selectbox("Choose the page", ["File Upload", "Question Answering"])

if page == "Question Answering":
    st.header("RAG Question Answering")
    query = st.text_input("Enter your question")

    if st.button("Submit Query"):
        with st.spinner("Retrieving and generating answer..."):
            # Call the answer_question function from the retriever  
            answer, refs = answer_question(query)
        st.subheader("Answer")
        st.write(answer)
        
        with st.expander("Reference Sources"):
            for ref in refs:
                st.markdown(f"- {ref}")

elif page == "File Upload":
    st.header("File Upload & Indexing")
    uploaded_file = st.file_uploader("Please select a file", type=["pdf", "docx", "txt"])

    # if uploaded_file is not None:
    #     with st.spinner("Processing file..."):
    #         # Call the process_file function from the indexer
    #         result = process_file(uploaded_file)
    #     st.success(f"File processed and indexed: {uploaded_file.name}")
    #     st.json(result)  # Can show the chunk result or metadata                