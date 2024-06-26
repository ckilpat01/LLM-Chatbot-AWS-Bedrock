#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Creation Of Streamlit App


# In[ ]:


# Import Libraries
import streamlit as st
import uuid
import bedrock


# In[ ]:


# Configure Streamlit session_state
USER_ICON = "images/user-icon.png"
AI_ICON = "images/ai-icon.png"

if "user_id" in st.session_state:
    user_id = st.session_state["user_id"]
else:
    user_id = str(uuid.uuid4())
    st.session_state["user_id"] = user_id

if "llm_chain" not in st.session_state:
    st.session_state["llm_app"] = bedrock
    st.session_state["llm_chain"] = bedrock.bedrock_chain()

if "questions" not in st.session_state:
    st.session_state.questions = []

if "answers" not in st.session_state:
    st.session_state.answers = []

if "input" not in st.session_state:
    st.session_state.input = ""


# In[ ]:


# Function To Create Top Bar
def write_top_bar():
    col1, col2, col3 = st.columns([2, 10, 3])
    with col2:
        header = "Amazon Bedrock Chatbot"
        st.write(f"<h3 class='main-header'>{header}</h3>", unsafe_allow_html=True)
    with col3:
        clear = st.button("Clear Chat")

    return clear

# Button for clearing chat
clear = write_top_bar()

# Function with some clearing functionality
if clear:
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.input = ""
    bedrock.clear_memory(st.session_state["llm_chain"])


# In[ ]:


# Function for handling input from user
def handle_input():
    input = st.session_state.input

    llm_chain = st.session_state["llm_chain"]
    chain = st.session_state["llm_app"]
    result, amount_of_tokens = chain.run_chain(llm_chain, input)
    question_with_id = {
        "question": input,
        "id": len(st.session_state.questions),
        "tokens": amount_of_tokens,
    }
    st.session_state.questions.append(question_with_id)

    st.session_state.answers.append(
        {"answer": result, "id": len(st.session_state.questions)}
    )
    st.session_state.input = ""


# In[ ]:


# Call All Input Functions And Add Form

def write_user_message(md):
    col1, col2 = st.columns([1, 12])

    with col1:
        st.image(USER_ICON, use_column_width="always")
    with col2:
        st.warning(md["question"])
        st.write(f"Tokens used: {md['tokens']}")


def render_answer(answer):
    col1, col2 = st.columns([1, 12])
    with col1:
        st.image(AI_ICON, use_column_width="always")
    with col2:
        st.info(answer["response"])


def write_chat_message(md):
    chat = st.container()
    with chat:
        render_answer(md["answer"])


with st.container():
    for q, a in zip(st.session_state.questions, st.session_state.answers):
        write_user_message(q)
        write_chat_message(a)


st.markdown("---")
input = st.text_input(
    "You are talking to an AI, ask any question.", key="input", on_change=handle_input
)

