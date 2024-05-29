import streamlit as st
import openai
import json

# OpenAI API 키 설정
openai.api_key = 'your_openai_api_key'

def get_explanation(quiz, correct_answer):
    prompt = f"문제: {quiz}\n정답: {correct_answer}\n이 문제의 해설을 작성해 주세요."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    explanation = response.choices[0].message['content'].strip()
    return explanation

def quiz_grading_page():
    st.title("퀴즈 리뷰 페이지")
    st.markdown("---")
    
    # 세션 상태 초기화
    if 'number' not in st.session_state:
        st.session_state.number = 0
    if 'quizs' not in st.session_state or not st.session_state.quizs:
        st.warning("퀴즈가 없습니다. 먼저 퀴즈를 풀어주세요.")
        return
    
    # 현재 문제 번호가 유효한지 확인
    if st.session_state.number < 0 or st.session_state.number >= len(st.session_state.quizs):
        st.warning("유효하지 않은 문제 번호입니다.")
        return
    
    question = st.session_state.quizs[st.session_state.number]
    res = json.loads(question["answer"])
    
    st.header(f"문제 {st.session_state.number + 1}")
    st.write(f"**{res['quiz']}**")
    
    # 객관식 문제의 경우 선택지 출력
    if 'options1' in res and 'options2' in res:
        st.write(f"1. {res['options1']}")
        st.write(f"2. {res['options2']}")
        if 'options3' in res:
            st.write(f"3. {res['options3']}")
        if 'options4' in res:
            st.write(f"4. {res['options4']}")
    
    st.write(f"정답: {res['correct_answer']}")
    
    # explanation = get_explanation(res['quiz'], res['correct_answer'])
    # st.write(f"해설: {explanation}")
    # st.markdown("---")
    
    col1, col2= st.columns(2)
    with col1:
        if st.button("이전 문제"):
            if st.session_state.number > 0:
                st.session_state.number -= 1  # 이전 문제로 이동
            else:
                st.warning("첫 번째 문제입니다.")
    with col2:
        if st.button("다음 문제"):
            if st.session_state.number < len(st.session_state.quizs) - 1:
                st.session_state.number += 1  # 다음 문제로 이동
            else:
                st.warning("마지막 문제입니다.")

if __name__ == "__main__":
    quiz_grading_page()
