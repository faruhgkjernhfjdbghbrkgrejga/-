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
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import io
import json

class CreateQuizoub(BaseModel):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="The first option of the created problem")
    options2: str = Field(description="The second option of the created problem")
    options3: str = Field(description="The third option of the created problem")
    options4: str = Field(description="The fourth option of the created problem")
    correct_answer: str = Field(description="One of the options1 or options2 or options3 or options4")

def connect_db():
    client = MongoClient("mongodb+srv://username:password@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=YourApp")
    return client['your_database_name']

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

def search_vectors(collection_name, query_vector, top_k=10):
    db = connect_db()
    collection = db[collection_name]
    results = collection.aggregate([
        {
            '$search': {
                'vector': {
                    'query': query_vector,
                    'path': 'vector',
                    'cosineSimilarity': True,
                    'topK': top_k
                }
            }
        }
    ])
    return list(results)

def retrieve_results(user_query):
    # Create MongoDB Atlas Vector Search instance
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        "mongodb+srv://username:password@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=YourApp",
        "database.collection",
        OpenAIEmbeddings(model="gpt-3.5-turbo-0125"),
        index_name="vector_index"
    )

    # Perform vector search based on user input
    response = vector_search.similarity_search_with_score(
        input=user_query, k=5, pre_filter={"page": {"$eq": 1}}
    )

    # Check if any results are found
    if not response:
        return None

    return response

def generate_quiz(quiz_type, text_content, retrieval_chainoub):
    # Generate quiz prompt based on selected quiz type
    if quiz_type == "다중 선택 (객관식)":
        response = retrieval_chainoub.invoke(
            {
                "input": "Create one multiple-choice question focusing on important concepts, following the given format, referring to the following context"
            }
        )
    quiz_questions = response
    return quiz_questions

def process_file(uploaded_file, upload_option):
    if uploaded_file is None:
        st.warning("파일을 업로드하세요.")
        return None

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
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([text_content])
    return texts

def quiz_creation_page():
    st.title("AI 퀴즈 생성기")
    st.markdown("---")
    
    # 퀴즈 유형 선택
    quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])

    # 퀴즈 개수 선택
    num_quizzes = st.number_input("생성할 퀴즈의 개수를 입력하세요:", min_value=1, value=5, step=1)

    # 파일 업로드 옵션 선택
    upload_option = st.radio("입력 유형을 선택하세요", ("PDF 파일", "텍스트 파일", "URL", "토픽 선택"))

    # 파일 업로드 옵션
    st.header("파일 업로드")
    uploaded_file = None
    text_content = None
    topic = None

    if upload_option == "토픽 선택":
        topic = st.selectbox(
            "토픽을 선택하세요",
            ("수학", "문학", "비문학", "과학"),
            index=None,
            placeholder="토픽을 선택하세요",
        ) 

    elif upload_option == "URL":
        url_area_content = st.text_area("URL을 입력하세요.")
        loader = RecursiveUrlLoader(url=url_area_content)
        text_content = loader.load()
        
    else:
        text_content = process_file(uploaded_file, upload_option)
    
    quiz_questions = []

    if text_content is not None:
        if st.button('문제 생성 하기'):
            with st.spinner('퀴즈를 생성 중입니다...'):
                llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
                embeddings = OpenAIEmbeddings()

                # Rag
                text_splitter = RecursiveCharacterTextSplitter()
                documents = text_splitter.split_documents(text_content)
                vector = FAISS.from_documents(documents, embeddings)

                # PydanticOutputParser 생성
                parseroub = PydanticOutputParser(pydantic_object=CreateQuizoub)

                prompt = PromptTemplate.from_template(
                    "{input}, Please answer in KOREAN."

                    "CONTEXT:"
                    "{context}."

                    "FORMAT:"
                    "{format}"
                )
                promptoub = prompt.partial(format=parseroub.get_format_instructions())

                document_chainoub = create_stuff_documents_chain(llm, promptoub)

                retriever = vector.as_retriever()

                retrieval_chainoub = create_retrieval_chain(retriever, document_chainoub)

                for i in range(num_quizzes):
                    quiz_questions.append(generate_quiz(quiz_type, text_content, retrieval_chainoub))
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
