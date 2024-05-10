# main.py

import streamlit as st
from upload_page import upload_page
from quiz_creation_page import quiz_creation_page
from quiz_solve_page import quiz_solve_page
from quiz_grading_page import quiz_grading_page
from sign import sign

# 페이지 타이틀 설정
st.set_page_config(page_title="AI 퀴즈 생성기")

# 메인 함수
def main():
    # 사이드바 메뉴 설정
    menu_options = ["로그인", "파일 업로드", "퀴즈 생성", "퀴즈 풀이", "퀴즈 채점"]
    selected_page = st.sidebar.radio("메뉴", menu_options)

    # 선택된 페이지 표시
    if selected_page == "파일 업로드":
        text_content = upload_page()
        st.session_state.text_content = text_content
    elif selected_page == "퀴즈 생성":
        text_content = st.session_state.get("text_content", None)
        if text_content:
            quiz_creation_page(text_content)
        else:
            st.warning("먼저 파일을 업로드하세요.")
    elif selected_page == "퀴즈 풀이":
        quiz_solve_page()
    elif selected_page == "퀴즈 채점":
        quiz_grading_page()
    elif selected_page == "로그인":
        sign()

# 메인 함수 실행
if __name__ == "__main__":
    main()
