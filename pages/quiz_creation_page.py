import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import io
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

class CreateQuizoub(BaseModel):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="The first option of the created problem")
    options2: str = Field(description="The second option of the created problem")
    options3: str = Field(description="The third option of the created problem")
    options4: str = Field(description="The fourth option of the created problem")
    correct_answer: str = Field(description="One of the options1 or options2 or options3 or options4")

class CreateQuizsub(BaseModel):
    quiz: str = Field(description="The created problem")
    correct_answer: str = Field(description="The answer to the problem")

class CreateQuizTF(BaseModel):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="The true or false option of the created problem")
    options2: str = Field(description="The true or false option of the created problem")
    correct_answer: str = Field(description="One of the options1 or options2")

def make_model(pages):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    embeddings = OpenAIEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(pages)
    vector = FAISS.from_documents(documents, embeddings)

    parseroub = PydanticOutputParser(pydantic_object=CreateQuizoub)
    parsersub = PydanticOutputParser(pydantic_object=CreateQuizsub)
    parsertf = PydanticOutputParser(pydantic_object=CreateQuizTF)

    prompt = PromptTemplate.from_template(
        "Question: {input}, Please answer in KOREAN.\n\n"
        "CONTEXT:\n"
        "{context}.\n\n"
        "FORMAT:\n"
        "{format}"
    )
    promptoub = prompt.partial(format=parseroub.get_format_instructions())
    promptsub = prompt.partial(format=parsersub.get_format_instructions())
    prompttf = prompt.partial(format=parsertf.get_format_instructions())

    document_chainoub = create_stuff_documents_chain(llm, promptoub)
    document_chainsub = create_stuff_documents_chain(llm, promptsub)
    document_chaintf = create_stuff_documents_chain(llm, prompttf)

    retriever = vector.as_retriever()

    retrieval_chainoub = create_retrieval_chain(retriever, document_chainoub)
    retrieval_chainsub = create_retrieval_chain(retriever, document_chainsub)
    retrieval_chaintf = create_retrieval_chain(retriever, document_chaintf)

    return retrieval_chainoub, retrieval_chainsub, retrieval_chaintf

@st.cache_data
def process_file(uploaded_file, text_area_content=None, url_area_content=None):
    text_content = None

    if uploaded_file:
        if uploaded_file.type == "text/plain":
            text_content = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type.startswith("image/"):
            image = Image.open(uploaded_file)
            text_content = pytesseract.image_to_string(image)
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text()
        else:
            st.error("지원하지 않는 파일 형식입니다.")
            return None
    elif text_area_content:
        text_content = text_area_content
    elif url_area_content:
        loader = RecursiveUrlLoader(url=url_area_content)
        text_content = loader.load()

    if text_content:
        documents = [{"page_content": text_content}]
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(documents)
        return documents
    else:
        st.warning("파일, 텍스트 또는 URL을 입력하세요.")
        return None

@st.experimental_fragment
def generate_quiz(quiz_type, text_content, retrieval_chainoub, retrieval_chainsub, retrieval_chaintf):
    if quiz_type == "다중 선택 (객관식)":
        response = retrieval_chainoub.invoke(
            {
                "input": "Create one multiple-choice question focusing on important concepts, following the given format, referring to the following context",
                "context": text_content
            }
        )
    elif quiz_type == "주관식":
        response = retrieval_chainsub.invoke(
            {
                "input": "Create one open-ended question focusing on important concepts, following the given format, referring to the following context",
                "context": text_content
            }
        )
    elif quiz_type == "OX 퀴즈":
        response = retrieval_chaintf.invoke(
            {
                "input": "Create one true or false question focusing on important concepts, following the given format, referring to the following context",
                "context": text_content
            }
        )
    else:
        response = []

    return response

@st.experimental_fragment
def grade_quiz_answer(user_answer, quiz_answer):
    return "정답" if user_answer.lower() == quiz_answer.lower() else "오답"

def quiz_creation_page():
    st.title("AI 퀴즈 생성기")
    
    quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])
    num_quizzes = st.number_input("생성할 퀴즈의 개수를 입력하세요:", min_value=1, value=5, step=1)

    upload_option = st.radio("입력 유형을 선택하세요", ("이미지 파일", "PDF 파일", "직접 입력", "URL", "토픽 선택"))

    uploaded_file = None
    text_area_content = None
    url_area_content = None
    selected_topic = None

    if upload_option == "이미지 파일":
        uploaded_file = st.file_uploader("이미지 파일을 업로드하세요.", type=["jpg", "jpeg", "png"])
    elif upload_option == "PDF 파일":
        uploaded_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])
    elif upload_option == "직접 입력":
        text_area_content = st.text_area("텍스트를 입력하세요.")
    elif upload_option == "URL":
        url_area_content = st.text_area("URL을 입력하세요.")
    elif upload_option == "토픽 선택":
        selected_topic = st.selectbox(
            "토픽을 선택하세요.",
            ("토픽 선택", "수학", "물리학", "역사", "화학")
        )

    if upload_option in ["이미지 파일", "PDF 파일"]:
        text_content = process_file(uploaded_file)
    elif upload_option == "직접 입력":
        text_content = process_file(None, text_area_content)
    elif upload_option == "URL":
        text_content = process_file(None, None, url_area_content)
    else:
        st.warning("입력이 필요합니다.")
        return

    if text_content is not None:
        if st.button('문제 생성 하기'):
            with st.spinner('퀴즈를 생성 중입니다...'):
                retrieval_chainoub, retrieval_chainsub, retrieval_chaintf = make_model(text_content)
                quiz_questions = [generate_quiz(quiz_type, text_content, retrieval_chainoub, retrieval_chainsub, retrieval_chaintf) for _ in range(num_quizzes)]
                
                st.session_state['quiz_questions'] = quiz_questions
                st.session_state['quiz_created'] = True

                st.success('퀴즈 생성이 완료되었습니다!')
                st.write(quiz_questions)

    if st.session_state.get('quiz_created', False):
        if st.button('퀴즈 풀기'):
            st.write("퀴즈 풀기 기능을 구현하세요.")  # Placeholder for quiz solving page navigation

if __name__ == "__main__":
    quiz_creation_page()
