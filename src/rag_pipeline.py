from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def create_rag_pipeline(vectorstore):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatOllama(
        model="qwen2.5:1.5b",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an assistant that answers questions using the
    provided context.

    Context:
    {context}

    Question:
    {question}

    Instructions:
    - Answer only using the provided context.
    - Do not invent information.
    - If the context does not contain enough information,
      say: I don't have enough information to answer this question.
    """)

    def ask_question(question):

        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        messages = prompt.invoke({
            "context": context,
            "question": question
        })

        response = llm.invoke(messages)

        return {
            "answer": response.content,
            "sources": docs
        }

    return ask_question