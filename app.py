import streamlit as st
from langchain_aws import ChatBedrock
from langchain_community.vectorstores import Chroma
from langchain_aws import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.tools import tool
import yfinance as yf
from dotenv import load_dotenv
import os

load_dotenv()

@tool
def get_stock_price(ticker: str) -> str:
    """Get latest stock price for a ticker"""
    try:
        price = yf.Ticker(ticker).info['currentPrice']
        return f"The latest price of {ticker.upper()} is £{price}"
    except Exception as e:
        return f"Could not fetch price for {ticker}: {str(e)}"

@tool
def calculator(expression: str) -> str:
    """Simple calculator for math expressions"""
    try:
        allowed_names = {"__builtins__": {}}  # Secure eval
        result = eval(expression, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Invalid math expression: {str(e)}"

tools = [get_stock_price, calculator]

# Load vector DB
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 6})

# RAG chain
template = """You are a helpful financial regulation assistant. Use the context and tools when needed.

Context: {context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatBedrock(
    model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",  # Claude 3.7 Sonnet (native eu-west-2, agentic)
    streaming=True,
    region_name="eu-west-2"  # London region
)
# Fallback: If issues, change to "anthropic.claude-3-sonnet-20240229-v1:0" (Claude 3 Sonnet)
llm_with_tools = llm.bind_tools(tools)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm_with_tools | StrOutputParser()
)

st.title("Financial Regulation RAG Agent with Tool Calling")
st.write("Built with AWS Bedrock (Claude 3.7 Sonnet) + LangChain + Chroma | Nikhil Muneshwar")

if prompt := st.chat_input("Ask about FCA rules, PRA, or live stock prices..."):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        try:
            response = chain.stream(prompt)
            st.write_stream(response)
        except Exception as e:
            st.error(f"Error: {str(e)}. Check Bedrock use case submission.")