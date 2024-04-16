import streamlit as st
from langchain.chat_models import ChatOpenAI
from upload_page import process_file

chat_model = ChatOpenAI()

def generate_quiz(quiz_type, text_content):
    if quiz_type == "다중 선택 (객관식)":
        prompt = "객관식 퀴즈를 생성합니다."
    elif quiz_type == "주관식":
        prompt = "주관식 퀴즈를 생성합니다."
    else:
        prompt = "OX 퀴즈를 생성합니다."

    prompt += f'''
    다음 텍스트를 기반으로 퀴즈를 생성합니다:

    {text_content}

    다양한 유형의 문제를 포함하여 퀴즈를 생성하세요. 객관식, 주관식, OX 퀴즈 등을 포함하여 참가자의 이해도와 지식 깊이를 테스트하세요.
    '''

    quiz_questions = chat_model.predict(prompt)
    quiz_questions = quiz_questions.split("\n")

    return quiz_questions

def quiz_creation_page():
    st.title("퀴즈 생성")

    quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])
    uploaded_file = st.file_uploader("텍스트, 이미지, 또는 PDF 파일을 업로드하세요.", type=["txt", "jpg", "jpeg", "png", "pdf"])

    if uploaded_file is not None:
        text_content = process_file(uploaded_file)

        if text_content is not None:
            if st.button("퀴즈 생성하기"):
                quiz_questions = generate_quiz(quiz_type, text_content)

                st.header("생성된 퀴즈")
                for question in quiz_questions:
                    st.write(question)
