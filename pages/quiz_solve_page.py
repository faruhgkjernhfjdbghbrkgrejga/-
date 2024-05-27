#quiz_solve_page.py

import streamlit as st
import json

def quiz_solve_page():
    if 'quizs' not in st.session_state:
        # 예시 퀴즈 데이터
        st.session_state.quizs = [
            {"question": "파이썬의 창시자는?", "options": ["가이드 반 로섬", "제임스 고슬링", "데니스 리치", "마크 주커버그"], "answer": "가이드 반 로섬"},
            {"question": "HTML은 무엇의 약자인가?", "options": ["Hyper Trainer Marking Language", "Hyper Text Markup Language", "Home Tool Markup Language", "Hyperlinks and Text Markup Language"], "answer": "Hyper Text Markup Language"}
        ]

    if 'number' not in st.session_state:
        st.session_state.number = 0
    if 'user_selected_answers' not in st.session_state:
        st.session_state.user_selected_answers = []

    placeholder = st.empty()
    if st.session_state.number < len(st.session_state.quizs):
        question = st.session_state.quizs[st.session_state.number]
        with placeholder.container():
            st.header(f"질문 {st.session_state.number + 1}: {question['question']}")
            options = question['options']
            
            for index, option in enumerate(options):
                if st.button(f"{index+1}. {option}", key=f"{st.session_state.number}_{index}"):
                    st.session_state.user_selected_answers.append(option)
                    st.session_state.number += 1
                    placeholder.empty()
                    if st.session_state.number == len(st.session_state.quizs):
                        st.session_state.number = 0  # 모든 문제를 다 풀면 처음으로 돌아감
                        st.experimental_rerun()
                    else:
                        quiz_solve_page()
    else:
        st.write("모든 퀴즈를 완료했습니다!")
        st.write("사용자가 선택한 답변:")
        for i, answer in enumerate(st.session_state.user_selected_answers):
            correct_answer = st.session_state.quizs[i]['answer']
            if answer == correct_answer:
                st.write(f"질문 {i+1}: 정답입니다! ({answer})")
            else:
                st.write(f"질문 {i+1}: 오답입니다! (선택한 답: {answer}, 정답: {correct_answer})")
        if st.button("퀴즈 채점 페이지로 이동"):
            st.session_state.page = "quiz_grading_page"

if __name__ == "__main__":
    quiz_solve_page()

