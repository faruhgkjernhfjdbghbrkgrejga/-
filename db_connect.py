from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings

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

# #db_connect.py

# from langchain_community.vectorstores import MongoDBAtlasVectorSearch
# from langchain_openai import OpenAIEmbeddings

# def load_topic_data(topic):
#     # Create MongoDB Atlas Vector Search instance
#     vector_search = MongoDBAtlasVectorSearch.from_connection_string(
#         "mongodb+srv://username:password@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=YourApp",
#         "database.collection",
#         OpenAIEmbeddings(model="gpt-3.5-turbo-0125"),
#         index_name="vector_index"
#     )

#     # Perform vector search based on topic
#     response = vector_search.similarity_search_with_score(
#         input=topic, k=10, pre_filter={"topic": {"$eq": topic}}
#     )

#     # Check if any results are found
#     if not response:
#         return None

#     # Return the content of the retrieved documents
#     documents = [doc['content'] for doc in response]
#     return documents
