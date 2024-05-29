import streamlit as st

def quiz_grading_page():
    st.title("퀴즈 채점 페이지")
    st.markdown("---")

    # 세션 상태에서 퀴즈와 답변 가져오기
    quizs = st.session_state.get('quizs', [])
    user_selected_answers = st.session_state.get('user_selected_answers', [])
    correct_answers = st.session_state.get('correct_answers', [])
    number = st.session_state.get('number', 0)

    if not quizs:
        st.warning("퀴즈가 없습니다.")
        return

    # 현재 문제 출력
    current_quiz = quizs[number]
    st.header(f"문제 {number + 1}")
    st.write(f"**{current_quiz['quiz']}**")
    
    if 'options1' in current_quiz:
        st.write("선택지:")
        st.write(f"1. {current_quiz['options1']}")
        st.write(f"2. {current_quiz['options2']}")
        if 'options3' in current_quiz:
            st.write(f"3. {current_quiz['options3']}")
        if 'options4' in current_quiz:
            st.write(f"4. {current_quiz['options4']}")

    # 사용자가 입력한 답변 출력
    user_answer = user_selected_answers[number] if number < len(user_selected_answers) else "답변 없음"
    st.write(f"사용자 답변: {user_answer}")

    # 이전 문제, 다음 문제 버튼
    col1, col2 = st.columns(2)
    with col1:
        if st.button("이전 문제"):
            if number > 0:
                st.session_state.number -= 1
            else:
                st.warning("첫 번째 문제입니다.")
    with col2:
        if st.button("다음 문제"):
            if number < len(quizs) - 1:
                st.session_state.number += 1
            else:
                st.warning("마지막 문제입니다.")

if __name__ == "__main__":
    quiz_grading_page()

