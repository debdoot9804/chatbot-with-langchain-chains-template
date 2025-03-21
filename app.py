from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

load_dotenv()

st.set_page_config(page_title="Q&A demo")
st.header("Langchain Application using Groq with Chaining")

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


system_input = st.text_input("Please enter a system message for our LLM")
human_input = st.text_input("Enter your query")


def get_response(system_input, topic):
    llm = ChatGroq(model="llama-3.3-70b-versatile")

    
    chat_template = ChatPromptTemplate.from_messages([
        ("system", system_input),
        ("human", "Explain the {topic}.")
    ])

    
    chain = chat_template | llm #chaining

    
    result = chain.invoke({"topic": topic}).content
    return result


if st.button("Ask the Question"):
    if not system_input or not human_input:
        st.warning("Please enter both thr system message and alo the query.")
    else:
        response = get_response(system_input, human_input)
        st.subheader("The Response is:")
        st.write(response)
