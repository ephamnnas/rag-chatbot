import streamlit as st
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ai21.embeddings import AI21Embeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import ChatOllama

# Step 1: Upload PDF and process
st.title("PDF Query Chatbot")

# File uploader for the PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    try:
        pdf_data = BytesIO(uploaded_file.read())
        reader = PdfReader(pdf_data)
    except Exception as e:
        st.error(f"Failed to process file: {e}")
        st.stop()  # Prevent further execution if there's an error

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_text(text)

    # Create vector store from the document chunks
    vector_store = Chroma.from_texts(
        texts=all_splits,
        embedding=AI21Embeddings(api_key="cGXAuU1AMSRkgOvNBgtKfR1LH4JLE825")  
    )

    retriever = vector_store.as_retriever(k=2)

    def doc_retriever(retriever, query=''):
        docs = retriever.invoke(query)
        doc_texts = [doc.page_content for doc in docs]  
        return doc_texts


    # Step 2: Create the chat prompt template
    prompt_ = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that answers questions when you know the answer from the context. Humbly reply that you do not know the answer if you have no information in the context."),
            MessagesPlaceholder(variable_name='history', optional=True),
            MessagesPlaceholder(variable_name='context', optional=True),
            ("human", "{query}"),
        ]
    )

    # Step 3: Chatbot conversation logic
    chat_history = ChatMessageHistory()
    chat_history_limit = 4
    chat_model = ChatOllama(model="llama3.2:latest", temperature=0)

    def add_to_history(chat_history_demo, type='', message=''):
        if type == 'user':
            chat_history_demo.add_user_message(message=message)
        elif type == 'ai':
            chat_history_demo.add_ai_message(message=message)
        return chat_history_demo

    def chat_summarizer(chat_history, chat_model):
        memory = ConversationSummaryMemory.from_messages(
            llm=chat_model,
            chat_memory=chat_history,
            return_messages=True
        )
        summary = [memory.buffer]
        return summary

    # Step 4: Handle user input and query response
    query = st.text_input("Ask a question based on the PDF content")  

    if query:
        # Add user message to chat history
        chat_history = add_to_history(chat_history, type='user', message=query)
        
        # Retrieve relevant document texts
        context = doc_retriever(retriever=retriever, query=query)

        # If chat history is too long, summarize it
        if len(chat_history.messages) > chat_history_limit:
            summary = chat_summarizer(chat_history=chat_history, chat_model=chat_model)
            prompt = prompt_.invoke({'history': summary, 'context': context, 'query': query})
        else:
            prompt = prompt_.invoke({'history': chat_history.messages, 'context': context, 'query': query})

        # Get chatbot response
        response = chat_model(prompt.messages)
        
        # Display response in the app
        st.write("Chatbot: ", response.content)
        
        # Add chatbot response to chat history
        chat_history = add_to_history(chat_history, type='ai', message=response.content)

