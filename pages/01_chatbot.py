import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.storage import LocalFileStore
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
 
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        st.session_state["messages"].append({"message":self.message, "role": "ai"})
        
    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    streaming=True,
    callbacks=[
        ChatCallbackHandler()
        ]
    )

memory = ConversationBufferMemory(
    llm=llm, max_token_limit=120, memory_key="chat_history", return_messages=True
)

def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]

@st.cache_resource
def embed_files(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="¥n",
        chunk_size=400,
        chunk_overlap=50
    )
    loader = UnstructuredFileLoader(f"./.cache/files/{file.name}")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)
    vectorstore = Chroma.from_documents(
        docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message":message, "role": role})

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False
            )

def format_docs(docs):
    return "¥n¥n".join(document.page_content for document in docs)

final_prompt= ChatPromptTemplate.from_messages([
    (
        "system", 
        """
        Answer the question using ONLY the following content. If you don't know the answer,
        just say "sorry, I don't have answer". DON'T make anything up.
        
        Context: {context}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

st.title("Let's make a quiz🪄")

st.markdown(
    """
    Let's make a quiz from uploaded stroy.
    """)
with st.sidebar:
    file = st.file_uploader("upload a txt, docx or pdf files", 
                            type=["txt", "pdf", "docx"],
                            )
if file:
    retriever = embed_files(file)
    send_message("Ask anything", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file")

    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(load_memory)
            } | final_prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)
            memory.save_context(
                {"input": message},
                {"output": response.content})
else:
    st.session_state["messages"] = []