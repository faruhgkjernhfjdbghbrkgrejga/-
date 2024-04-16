import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

@st.cache(allow_output_mutation=True)
def grade_quiz_answers(user_answers, quiz_answers):
    graded_answers = []
    for user_answer, quiz_answer in zip(user_answers, quiz_answers):
        similarity = cosine_similarity([user_answer], [quiz_answer])[0][0]
        if similarity > 0.7:
            graded_answers.append("정답")
        else:
            graded_answers.append("오답")
    return graded_answers

def quiz_grading_page():
    st.title("퀴즈 채점하기")

    st.write("퀴즈 채점 페이지의 기능 구현은 여기에 작성합니다.")
