from langchain_huggingface import HuggingFaceEmbeddings

def load_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small"
    )
    return embeddings