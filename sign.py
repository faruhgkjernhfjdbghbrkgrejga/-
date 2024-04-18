import streamlit as st

# 샘플 사용자 데이터베이스 (실제로는 데이터베이스나 API 호출로 대체해야 합니다)
users = {
    "user1": {"password": "password1"},
    "user2": {"password": "password2"}
}

def register_user(username, password):
    """
    새로운 사용자를 등록합니다.

    Args:
        username (str): 원하는 사용자 이름.
        password (str): 원하는 비밀번호.

    Returns:
        str: 성공적인 등록 또는 오류 메시지.
    """
    if username in users:
        return f"오류: '{username}' 사용자가 이미 존재합니다."
    else:
        users[username] = {"password": password}
        return f"'{username}' 사용자가 성공적으로 등록되었습니다."

def login_user(username, password):
    """
    기존 사용자를 로그인합니다.

    Args:
        username (str): 사용자 이름.
        password (str): 비밀번호.

    Returns:
        str: 성공적인 로그인 또는 오류 메시지.
    """
    if username in users and users[username]["password"] == password:
        return f"'{username}' 사용자가 성공적으로 로그인되었습니다."
    else:
        return "오류: 잘못된 사용자 이름 또는 비밀번호입니다."

def main():
    st.title("User Registration & Login")

    # User registration
    st.header("Register")
    new_username = st.text_input("Enter a new username:")
    new_password = st.text_input("Enter a new password:", type="password")
    if st.button("Register"):
        result = register_user(new_username, new_password)
        st.success(result)

    # User login
    st.header("Login")
    existing_username = st.text_input("Enter your username:")
    existing_password = st.text_input("Enter your password:", type="password")
    if st.button("Login"):
        result = login_user(existing_username, existing_password)
        st.success(result)

if __name__ == "__main__":
    main()
