import streamlit as st
import os

st.set_page_config(
    page_title="AssistantGPT",
    page_icon="🍵",
)

with st.sidebar:
    OPENAI_API_KEY = ""
    OPENAI_API_KEY = st.text_input("Write your OPEN API KEY")
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"]
    gitLink = st.link_button(label="Go to Github Repo", url="https://github.com/sssssshhh/myChatGPT/blob/main/pages/04_Assistant.py" )
    streamlitLink = st.link_button(label="Go to the Streamlit App", url="https://myfirstaichatbot.streamlit.app")

def read_file_content(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    else:
        return "Please check the file path again"

def split_by_blank_lines(content):
    sections = content.split('\n\n')
    sections = [section.strip() for section in sections if section.strip()]
    return sections

def send_message(message, role):
    with st.chat_message(role):
        st.markdown(message)

file_path = "output.txt"

content = read_file_content(file_path)

if content:
    messages = split_by_blank_lines(content)
    for message in messages:
        if "user: " in message:
            send_message(message.replace("user: ", ""), "user")
        else:
            send_message(message, "ai")
