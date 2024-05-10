# main.py
import streamlit as st
from upload_page import upload_page
import quiz_creation_page
from quiz_solve_page import quiz_solve_page
from quiz_grading_page import quiz_grading_page
from sign import sign

def main():
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "파일 업로드"

    # 선택된 페이지 표시
    if selected_page == "파일 업로드":
        upload_page()
    elif selected_page == "퀴즈 생성":
        quiz_creation_page.quiz_creation_page()
    elif selected_page == "퀴즈 풀이":
        if st.button("퀴즈 풀기"):
            st.switch_page("quiz_solve_page")
    elif selected_page == "퀴즈 채점":
        quiz_grading_page()
    elif selected_page == "로그인":
        sign()

if __name__ == "__main__":
    main()
