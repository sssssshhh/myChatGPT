import json
import os
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser, output_parser

st.set_page_config(
    page_title="QuizGPT",
    page_icon="📚"
    )

st.title("QuizGPT")

create_quiz_function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
    ).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        create_quiz_function,
    ],
)

def format_docs(docs):
    return "¥n¥n".join(document.page_content for document in docs)

easy_question_prompt= ChatPromptTemplate.from_messages([
    (
        "system", 
        """
    You are a helpful assistant that is role playing as a teacher. 
    Your student is 4-6 year-old.
         
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    It should be very easy.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut

    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
        """),
    ])

difficult_question_prompt= ChatPromptTemplate.from_messages([
    (
        "system", 
        """
    You are a helpful assistant that is role playing as a professor. 
    Your student is over 20~30.
         
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    It should be tricky and difficult.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut

    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
        """),
    ])

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!
    Questions: {context}
""",
        )
    ]
)

formatting_chain = formatting_prompt | llm

def make_quiz_by_level(_level):
    if _level == "Easy":
        questions_chain = {"context": format_docs} | easy_question_prompt | llm
    else:
        questions_chain = {"context": format_docs} | difficult_question_prompt | llm
    return questions_chain

@st.cache_resource
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="¥n",
        chunk_size=400,
        chunk_overlap=50
    )
    loader = UnstructuredFileLoader(f"./.cache/quiz_files/{file.name}")
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, level, topic):
    chain = {"context": make_quiz_by_level(level)} | formatting_chain
    response = chain.invoke(_docs)
    response = response.additional_kwargs["function_call"]["arguments"]
    return response

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=2)
    docs = retriever.get_relevant_documents(term)
    return docs

with st.sidebar:
    OPENAI_API_KEY = ""
    OPENAI_API_KEY = st.text_input("Write your OPEN API KEY")
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"]
    link = st.link_button(label="Go to Github Repo", url="https://github.com/sssssshhh/myChatGPT" )

    docs = None
    topic = None
    
    choice = st.selectbox(
        "Choose what you want to use",
        ("File", "Wikepedia Article"))

    if choice == "File":
        file = st.file_uploader(
            "upload a txt, docx or pdf files", 
            type=["txt", "pdf", "docx"],
            )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia")
        if topic:
            docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
        Welcome to ChatGPT

        I will make a quiz for you.
        Get started by uploading a file or searching on Wikipedia in th sidebar
        """
        )
else:
    level = st.radio(
            "Choose the level",
            ["Easy", "Hard"],
            horizontal=True,
            index=None,
        )
    
    if st.session_state.get(f"quiz_completed_{level}_{topic or file.name}", False):
        st.success("Congratulations! You've successfully completed the quiz. 🎉")
        st.balloons()
    else:
        if level:
            response = run_quiz_chain(docs, level, topic if topic else file.name)
            correct_answers_count = 0

            with st.form("question_form_{level}_{topic}"):
                for question in json.loads(response)["questions"]:
                    st.write(question["question"])
                    value = st.radio(
                        "Select an option.",
                        [answer["answer"] for answer in question["answers"]],
                        index=None,
                    )

                    if {"answer":value, "correct": True} in question["answers"]:
                        st.success("Correct!")
                        correct_answers_count += 1
                    elif value is not None:
                        st.error("Incorrect..")
                button = st.form_submit_button()
                if button:
                    if correct_answers_count == len(json.loads(response)["questions"]):
                        st.session_state[f"quiz_completed_{level}_{topic or file.name}"] = True
                        st.rerun()
                       