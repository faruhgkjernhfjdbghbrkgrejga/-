import streamlit as st
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import io

# 파일 처리 함수
def process_file(uploaded_file):
    if uploaded_file is None:
        st.warning("파일을 업로드하세요.")
        return None

    # 업로드된 파일 처리
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

    return text_content

# 파일 업로드 페이지
def upload_page():
    st.title("파일 업로드")
    
    # 파일 업로드 옵션
    uploaded_file = st.file_uploader("텍스트, 이미지, 또는 PDF 파일을 업로드하세요.", type=["txt", "jpg", "jpeg", "png", "pdf"])

    # 파일 처리
    text_content = process_file(uploaded_file)

    if text_content is not None:
        st.success("파일 처리 완료!")

        # 업로드된 파일 내용 표시
        st.header("파일 내용")
        st.text(text_content)

# 테스트용 코드: upload_page() 함수 직접 호출
if __name__ == "__main__":
    upload_page()
