# import nest_asyncio
# nest_asyncio.apply()
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# from langchain.document_loaders.sitemap import SitemapLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document

import os
import time
from dotenv import load_dotenv

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain.vectorstores import Pinecone
from pinecone import Pinecone

import json
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# # Function to push embedded data to Vector Store - Pinecone
# def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):
#     text_splitter = RecursiveCharacterTextSplitter()
#     document_chunks = text_splitter.split_documents(docs)
#     pinecone = Pinecone(
#         api_key=pinecone_apikey,environment=pinecone_environment
#         )
#     # create a vectorstore from the chunks
#     vector_store=PineconeStore.from_documents(document_chunks,embeddings,index_name=pinecone_index_name)

def get_vectorstore():
    vector_store = PineconeStore.from_existing_index(index_name="reserch",embedding=embeddings)
    return vector_store

# def pull_from_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):
#     print("10secs delay...")
#     time.sleep(10)
#     pinecone = Pinecone(
#         api_key=pinecone_apikey,environment=pinecone_environment
#     )
#     index_name = pinecone_index_name
#     index = PineconeStore.from_existing_index(index_name, embeddings)
#     return index

# Function to find similar documents in Pinecone
# def similar_docs(query,pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,k=1):
#     pinecone = Pinecone(
#         api_key=pinecone_apikey,environment=pinecone_environment
#     )
#     index_name = pinecone_index_name
#     index = pull_from_pinecone(pinecone_apikey,pinecone_environment,index_name,embeddings)
    
#     index_stat = pinecone.Index(pinecone_index_name) 
#     vector_count = index_stat.describe_index_stats() 
#     k = vector_count["total_vector_count"]
    
#     similar_docs = index.similarity_search_with_score(query, int(k))
#     # Regular expression to extract metadata
#     metadata_pattern = r"metadata=\{'file': '(.*?)', 'folder': '(.*?)'\}"

#     # Search for metadata in page_content
#     metadata_match = re.search(metadata_pattern, page_content)

#     # Extract metadata if found
#     if metadata_match:
#         file_name = metadata_match.group(1)
#         folder_name = metadata_match.group(2)
        
#         metadata = {
#             'file': file_name,
#             'folder': folder_name
#         }
        
#         print(metadata)
#     else:
#         print("Metadata not found.")

#     print(doc_id)
#     return similar_docs

# def get_metadata()
#     metadatas = [{"page": i} for i in range(len(texts))]
#     docsearch = Pinecone.from_texts(
#         texts,
#         embedding_openai,
#         index_name=index_name,
#         metadatas=metadatas,
#         namespace=namespace_name,
#     )
#     output = docsearch.similarity_search(needs, k=1, namespace=namespace_name)
#     assert output == [Document(page_content=needs, metadata={"page": 0.0})]

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()  
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    # If there is no chat_history, then the input is just passed directly to the retriever. 
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    # for passing a list of Documents to a model.
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    print("getting response")
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    print(response['answer'])
    return response['answer']

st.set_page_config(page_title="Chat with Your Pages", page_icon="🤖")

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I can help you to query your Papers"),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore()  

# Render selected section
st.header('Enter your query')
# conversation
user_query = st.chat_input("Ask your query here...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    # sessions for code reload
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))
    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)