#quiz_creation_page.py

import streamlit as st
from pymongo import MongoClient, UpdateOne
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import io
import json
from pydantic import BaseModel, Field

class CreateQuiz(BaseModel):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="The first option of the created problem")
    options2: str = Field(description="The second option of the created problem")
    options3: str = Field(description="The third option of the created problem")
    options4: str = Field(description="The fourth option of the created problem")
    correct_answer: str = Field(description="One of the options1 or options2 or options3 or options4")

def connect_db():
    client = MongoClient("mongodb+srv://username:password@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=YourApp")
    return client['your_database_name']

def insert_documents(collection_name, documents):
    db = connect_db()
    collection = db[collection_name]
    collection.insert_many(documents)

def vectorize_and_store(data, collection_name):
    embeddings = OpenAIEmbeddings(api_key='your_openai_api_key')
    vector_operations = []

    for document in data:
        text = document['text']
        vector = embeddings.embed_text(text)
        operation = UpdateOne({'_id': document['_id']}, {'$set': {'vector': vector.tolist()}})
        vector_operations.append(operation)

    db = connect_db()
    collection = db[collection_name]
    collection.bulk_write(vector_operations)

def generate_quiz(quiz_type, text_content, vector_search):
    # Perform vector search based on text content
    response = vector_search.similarity_search_with_score(input=text_content, k=5)

    # Check if any results are found
    if not response:
        return None

    # Use the retrieved documents to create a quiz
    documents = [doc['content'] for doc in response]
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    embeddings = OpenAIEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(documents)
    vector = FAISS.from_documents(documents, embeddings)

    parser = PydanticOutputParser(pydantic_object=CreateQuiz)
    prompt = PromptTemplate.from_template(
        "Question: {input}, Please answer in KOREAN.\n\nCONTEXT:\n{context}.\n\nFORMAT:\n{format}"
    )
    prompt = prompt.partial(format=parser.get_format_instructions())

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke(
        {
            "input": f"Create one {quiz_type} question focusing on important concepts, following the given format, referring to the following context"
        }
    )

    return response

def quiz_creation_page():
    st.title("AI 퀴즈 생성기")
    st.markdown("---")

    quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])
    num_quizzes = st.number_input("생성할 퀴즈의 개수를 입력하세요:", min_value=1, value=5, step=1)
    upload_option = st.radio("입력 유형을 선택하세요", ("PDF 파일", "텍스트 파일", "URL", "토픽 선택"))

    text_content = None
    if upload_option == "텍스트 파일":
        uploaded_file = st.file_uploader("텍스트 파일을 업로드하세요.", type=["txt"])
        if uploaded_file is not None:
            text_content = uploaded_file.read().decode("utf-8")
    elif upload_option == "이미지 파일":
        uploaded_file = st.file_uploader("이미지 파일을 업로드하세요.", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            text_content = pytesseract.image_to_string(image)
    elif upload_option == "PDF 파일":
        uploaded_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])
        if uploaded_file is not None:
            pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text()
    elif upload_option == "URL":
        url_area_content = st.text_area("URL을 입력하세요.")
        if url_area_content:
            loader = RecursiveUrlLoader(url=url_area_content)
            text_content = loader.load()
    elif upload_option == "토픽 선택":
        topic = st.selectbox("토픽을 선택하세요", ("수학", "문학", "비문학", "과학"))
        if topic:
            text_content = topic

    if text_content:
        if st.button('문제 생성 하기'):
            with st.spinner('퀴즈를 생성 중입니다...'):
                vector_search = MongoDBAtlasVectorSearch.from_connection_string(
                    "mongodb+srv://username:password@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=YourApp",
                    "database.collection",
                    OpenAIEmbeddings(model="gpt-3.5-turbo-0125"),
                    index_name="vector_index"
                )

                quiz_questions = []
                for _ in range(num_quizzes):
                    quiz = generate_quiz(quiz_type, text_content, vector_search)
                    if quiz:
                        quiz_questions.append(quiz)

                st.session_state['quizs'] = quiz_questions
                st.session_state.selected_page = "퀴즈 풀이"
                st.session_state.selected_type = quiz_type
                st.session_state.selected_num = num_quizzes

                st.success('퀴즈 생성이 완료되었습니다!')
                st.write(quiz_questions)
                st.session_state['quiz_created'] = True

    if st.session_state.get('quiz_created', False):
        if st.button('퀴즈 풀기'):
            st.switch_page("pages/quiz_solve_page.py")

if __name__ == "__main__":
    quiz_creation_page()
