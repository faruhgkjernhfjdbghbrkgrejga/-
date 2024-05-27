#quiz_solve_page.py

import streamlit as st
from langchain_core.pydantic_v1 import BaseModel, Field
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import io
import json


def quiz_solve_page():
    placeholder = st.empty()
    if 'number' not in st.session_state:
        st.session_state.number = 0
    if 'user_selected_answers' not in st.session_state:
        st.session_state.user_selected_answers = []  # 사용자 선택 답변을 저장할 배열 초기화

    for j, question in enumerate(st.session_state.quizs):
        if st.session_state.number == j:
            with placeholder.container():
                res = json.loads(question["answer"])
                st.header(f"질문 {j+1}")
                st.write(f"{question}")
                options = [res.get('options1'), res.get('options2'), res.get('options3'), res.get('options4')]
                
                for index, option in enumerate(options):
                    if st.button(f"{index+1}. {option}", key=f"{j}_{index}"):
                        st.session_state.user_selected_answers.append(option)  # 선택한 답변을 배열에 추가
                        st.session_state.number += 1  # 다음 문제로 이동
                        if st.session_state.user_selected_answers.append == st.session_state.quizs[j]['answer']:
                            total_score += 1
                        # if st.session_state.number == len(st.session_state.quizs):
                        #     if st.button('퀴즈 채점'):
                        #         st.session_state['total_score'] = st.session_state.number  # 점수를 세션 상태에 저장
                        #         st.switch_page("pages/quiz_grading_page.py")
                            # st.session_state.number = 0  # 모든 문제를 다 풀면 처음으로 돌아감
                            # st.experimental_rerun()  # 페이지 새로고침
    
    if st.session_state.number == st.session_state.selected_num:
        if st.button('퀴즈 채점'):
            st.session_state['total_score'] = st.session_state.number  # 점수를 세션 상태에 저장
            st.switch_page("pages/quiz_grading_page.py")

    # 사용자가 선택한 답변 출력
    if st.session_state.user_selected_answers:
        st.write("사용자가 선택한 답변:")
        for answer in st.session_state.user_selected_answers:
            st.write(answer)

if __name__ == "__main__":
    quiz_solve_page()


