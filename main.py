import streamlit as st
from upload_page import upload_page
import quiz_creation_page
from quiz_solve_page import quiz_solve_page
from quiz_grading_page import quiz_grading_page
from sign import sign

def main():
    # 사이드바 메뉴 설정
    menu_options = ["로그인", "파일 업로드", "퀴즈 생성", "퀴즈 풀이", "퀴즈 채점"]
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "파일 업로드"
    selected_page = st.sidebar.radio("메뉴", menu_options, index=menu_options.index(st.session_state.selected_page))

    st.page_link("quiz_creation_page", label="퀴즈 생성", icon="🏠")
    st.page_link("page_1", label="Page 1", icon="1️⃣")
    st.page_link("page_2", label="Page 2", icon="2️⃣", disabled=True)
    st.page_link("http://www.google.com", label="Google", icon="🌎")

    # 선택된 페이지 표시
    if selected_page == "파일 업로드":
        upload_page()
    elif selected_page == "퀴즈 생성":
        quiz_creation_page.quiz_creation_page()
    elif selected_page == "퀴즈 풀이":
        quiz_solve_page()
    elif selected_page == "퀴즈 채점":
        quiz_grading_page()
    elif selected_page == "로그인":
        sign()

if __name__ == "__main__":
    main()
