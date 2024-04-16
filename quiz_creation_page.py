import streamlit as st
from langchain.chat_models import ChatOpenAI
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import io

# ChatOpenAI 모델 초기화
chat_model = ChatOpenAI()

# 파일 처리 함수
def process_file(uploaded_file):
    if uploaded_file is None:
        st.warning("파일을 업로드하세요.")
        return None

    # 업로드된 파일 처리
    if uploaded_file.type == "text/plain":
        text_content = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        text_content = pytesseract.image_to_string(image)
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text()
    else:
        st.error("지원하지 않는 파일 형식입니다.")
        return None

    return text_content

# 퀴즈 생성 함수
def generate_quiz(quiz_type, text_content):
    # Generate quiz prompt based on selected quiz type
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

    # Generate quizzes using ChatOpenAI model
    quiz_questions = chat_model.predict(prompt)

    # Convert quiz_questions to a list
    quiz_questions = quiz_questions.split("\n")

    return quiz_questions

# 퀴즈 생성 페이지
def quiz_creation_page():
    # Define quiz_questions variable
    quiz_questions = st.session_state.get("quiz_questions", [])

    st.title("퀴즈 생성 페이지")

    # 현재 페이지 상태
    page = st.session_state.get("page", 1)

    if page == 1:
        # 퀴즈 유형 선택
        quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])

        # 파일 업로드 옵션
        st.header("파일 업로드")
        uploaded_file = st.file_uploader("텍스트, 이미지, 또는 PDF 파일을 업로드하세요.", type=["txt", "jpg", "jpeg", "png", "pdf"])

        text_content = process_file(uploaded_file)

        if text_content is not None:
            if st.button("퀴즈 생성하기"):
                quiz_questions = generate_quiz(quiz_type, text_content)

                # Display generated quiz
                st.header("생성된 퀴즈")
                for question in quiz_questions:
                    st.write(question)

                # Move to page 2
                st.session_state["page"] = 2

    # Save quiz questions in session state
    if page == 1 and st.session_state.get("quiz_questions") != quiz_questions:
        st.session_state["quiz_questions"] = quiz_questions
