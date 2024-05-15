import streamlit as st
import upload_page
import quiz_creation_page
import quiz_solve_page
import quiz_grading_page
import sign

def main():
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "퀴즈 생성"
    selected_page = st.sidebar.radio("메뉴", menu_options, index=menu_options.index(st.session_state.selected_page))

    # 선택된 페이지 표시
    if selected_page == "퀴즈 생성":
        quiz_creation_page.quiz_creation_page()
    elif selected_page == "퀴즈 풀이":
        quiz_solve_page.quiz_solve_page()
    elif selected_page == "퀴즈 채점":
        quiz_grading_page.quiz_grading_page()
    elif selected_page == "로그인":
        sign.sign()

if __name__ == "__main__":
    main()
