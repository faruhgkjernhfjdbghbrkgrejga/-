import streamlit as st
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import io
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
import chardet
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from pymongo import MongoClient
import pymongo
import json
import openai


def get_explanation(quiz, correct_answer):
    prompt = f"문제: {quiz}\n정답: {correct_answer}\n이 문제의 해설을 작성해 주세요."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    explanation = response.choices[0].message['content'].strip()
    return explanation


def quiz_grading_page():
    st.title("퀴즈 리뷰 페이지")
    st.markdown("---")
    
    # 세션 상태 초기화
    if 'number' not in st.session_state:
        st.session_state.number = 0
    if 'quizs' not in st.session_state or not st.session_state.quizs:
        st.warning("퀴즈가 없습니다. 먼저 퀴즈를 풀어주세요.")
        return
    
    if st.session_state.number >= len(st.session_state.quizs):
        st.warning("유효하지 않은 문제 번호입니다.")
        return
    
    question = st.session_state.quizs[st.session_state.number]
    res = json.loads(question["answer"])
    
    st.header(f"문제 {st.session_state.number + 1}")
    st.write(f"**{res['quiz']}**")
    st.write(f"정답: {res['correct_answer']}")
    
    explanation = get_explanation(res['quiz'], res['correct_answer'])
    st.write(f"해설: {explanation}")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("이전 문제"):
            if st.session_state.number > 0:
                st.session_state.number -= 1  # 이전 문제로 이동
            else:
                st.warning("첫 번째 문제입니다.")
    with col2:
        if st.button("다음 문제"):
            if st.session_state.number < len(st.session_state.quizs) - 1:
                st.session_state.number += 1  # 다음 문제로 이동
            else:
                st.warning("마지막 문제입니다.")
    with col3:
        if st.button('퀴즈 풀이 페이지로 돌아가기'):
            st.switch_page("pages/quiz_solve_page.py")

if __name__ == "__main__":
    quiz_grading_page()
