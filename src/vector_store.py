from langchain_community.vectorstores import FAISS

def save_vectorstore(vectorstore, path="faiss_index"):
    vectorstore.save_local(path)

def load_vectorstore(embeddings, path="faiss_index"):
    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore