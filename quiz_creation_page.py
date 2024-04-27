def quiz_creation_page():
    # 퀴즈 유형 선택
    quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])

    # 퀴즈 개수 선택
    num_quizzes = st.number_input("생성할 퀴즈의 개수를 입력하세요:", min_value=1, value=5, step=1)

    # 파일 업로드 옵션
    st.header("파일 업로드")
    uploaded_file = st.file_uploader("텍스트, 이미지, 또는 PDF 파일을 업로드하세요.", type=["txt", "jpg", "jpeg", "png", "pdf"])

    # 텍스트 입력 옵션
    st.header("직접 텍스트 입력")
    text_input = st.text_area("텍스트를 입력하세요.")

    # 업로드된 파일 또는 직접 입력한 텍스트를 처리
    if uploaded_file is not None:
        text_content = process_file(uploaded_file)
    elif text_input:
        text_content = text_input
    else:
        st.warning("텍스트나 파일을 입력하세요.")
        return

    quiz_questions = []
    quiz_answers = []
    user_answers = []

    if text_content is not None:
        if 'quizs' not in st.session_state:
            st.session_state.quizs = None
        if st.button('문제 생성 하기'):
            for i in range(num_quizzes):
                quiz_questions.append(generate_quiz(quiz_type, text_content))
                quiz_answers.append(quiz_questions[i].correct_answer)
            st.session_state['quizs'] = quiz_questions
            st.session_state['canswer'] = quiz_answers
            if st.button("퀴즈 채점 페이지로 이동"):  # New button
                st.experimental_rerun()

        if st.session_state.quizs is not None:
            st.header("생성된 퀴즈")
            for j, question in enumerate(st.session_state.quizs):
                if quiz_type == "주관식":
                    st.write(f"주관식 문제{j+1}: {question.quiz}")
                    st.write("\n")
                else:
                    if quiz_type == "다중 선택 (객관식)":
                        st.write(f"객관식 문제{j+1}: {question.quiz}")
                    else:
                        st.write(f"OX퀴즈 문제{j+1}: {question.quiz}")
                    st.write("\n")
                    st.write(f"{question.options}")
                    st.write("\n")
                    st.write(f"여기는 퀴즈의 구조입니다: {question}")
                    st.write("\n")
                user_answer = st.text_input(f"질문{j + 1}에 대한 답변 입력", "1")
                user_answers.append(user_answer)
                st.session_state['uanswer'] = user_answers
                j += 1
                st.write("-----------------------------------------")
                st.write("\n")
