import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# 퀴즈 채점 함수
def grade_quiz_answers(user_answers, quiz_answers):
    graded_answers = []
    for user_answer, quiz_answer in zip(user_answers, quiz_answers):
        similarity = cosine_similarity([user_answer], [quiz_answer])[0][0]
        if similarity > 0.7:
            graded_answers.append("정답")
        else:
            graded_answers.append("오답")
    return graded_answers

# 퀴즈 채점 페이지 함수
def quiz_grading_page():
    # 현재 페이지 상태
    page_state = st.session_state

    if page_state["page"] == 1:
        # 퀴즈 채점 페이지
        st.title("퀴즈 채점하기")

        # Collect user answers
        st.header("답변 입력")
        user_answers = []
        for i in range(len(page_state["quiz_questions"])):
            user_answer = st.text_input(f"질문 {i+1}에 대한 답변 입력", "")
            user_answers.append(user_answer)

        # Button to grade the quiz
        if st.button("퀴즈 채점하기"):
            # Grade the quiz answers
            quiz_answers = [answer.split(": ")[1] for answer in page_state["quiz_questions"]]
            graded_answers = grade_quiz_answers(user_answers, quiz_answers)

            # Display grading results
            st.header("퀴즈 채점 결과")
            for i, (question, user_answer, graded_answer) in enumerate(zip(page_state["quiz_questions"], user_answers, graded_answers), start=1):
                st.write(f"질문 {i}: {question}")
                st.write(f"사용자 답변: {user_answer}")
                st.write(f"채점 결과: {graded_answer}")

            # Reset page state
            page_state["page"] = 1

