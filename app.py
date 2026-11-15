import streamlit as st

from src.embeddings import load_embeddings
from src.vector_store import load_vectorstore
from src.rag_pipeline import create_rag_pipeline


st.set_page_config(
    page_title="Assistant ai ,ml ,python,rag",
    page_icon="Hiba",
    layout="wide"
)

st.title(" Assistant ai ,ml ,python,rag")

st.write(
    "Ask questions and get answers based on your documents."
)


@st.cache_resource
def initialize_rag():

    embeddings = load_embeddings()

    vectorstore = load_vectorstore(embeddings)

    rag = create_rag_pipeline(vectorstore)

    return rag


try:
    rag = initialize_rag()

    question = st.text_input(
        "Enter your question:"
    )

    if st.button("Send"):

        if question.strip():

            with st.spinner("Generating answer..."):

                result = rag(question)

            st.subheader("Answer")

            st.write(result["answer"])

            st.subheader("Retrieved documents")

            for i, doc in enumerate(result["sources"]):

                with st.expander(f"Source {i+1}"):

                    st.write(doc.page_content)
                    st.write("Metadata:", doc.metadata)

        else:
            st.warning("Please enter a question.")

except Exception as e:
    st.error(f"An error occurred: {e}")