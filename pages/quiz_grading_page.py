#quiz_grading_page.py
import streamlit as st
from quiz_creation_page import quiz_questions
import json

def quiz_grading_page():
    # 세션 상태에서 사용자 답안과 정답 가져오기
    user_answers = st.session_state.get('user_answers', [])
    correct_answers = st.session_state.get('correct_answers', [])
    total_score = 0

    # 퀴즈 채점
    graded_answers = grade_quiz_answers(user_answers, correct_answers)

    results = []
    for user_answer, correct_answer in zip(user_answers, correct_answers):
        if user_answer == correct_answer:
            total_score += 1
            results.append('정답')
        else:
            results.append('오답')

    # 채점 결과 표시
    st.title("퀴즈 채점 결과")
    # total_score = 0
    for i, result in enumerate(graded_answers, start=1):
        st.subheader(f"문제 {i}")
        st.write(f"사용자 답변: {user_answers[i-1]}")
        st.write(f"정답: {correct_answers[i-1]}")
        if result == "정답":
            st.success("정답입니다!")
            total_score += 1
        else:
            st.error("오답입니다.")
    st.session_state['total_score'] = total_score

    # 점수 결과 표시
    st.write(f"당신의 점수는 {st.session_state['total_score']}점 입니다.")

    # 퀴즈 생성 페이지로 이동 버튼
    if st.button("퀴즈 생성 페이지로 이동"):
        st.session_state["page"] = "quiz_creation_page"

if __name__ == "__main__":
    quiz_grading_page()
