import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
import tempfile
from langchain_core.messages import AIMessage, HumanMessage
from utils import get_model_response

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

def main():
    st.title("Chat with CSV using Gemini Pro")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file",type="csv")

    if uploaded_file is not None:
        print("Uploaded File",uploaded_file)
        # use tempfile because CSVLoader only accepts a file_path
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            print("Temp File path",tmp_file_path)

            # Initializing CSV_Loader
            csv_loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
            'delimiter': ','})

            # Load data into csv Loader
            data = csv_loader.load()

            # Initialize chat Interface
            user_query = st.chat_input("Ask Your Question:")
            if user_query is not None and user_query != "":
                response = get_model_response(data, user_query)
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=response))
    
    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


if __name__ == "__main__":
    main()
