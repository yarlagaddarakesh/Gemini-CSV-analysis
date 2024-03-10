from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS

genai.configure(api_key="AIzaSyBJdSOqDwQYlpoBQ4Mt-sP33fO_R8qPDQw")
os.environ["GOOGLE_API_KEY"]="AIzaSyBJdSOqDwQYlpoBQ4Mt-sP33fO_R8qPDQw"


#Function to get response from GEMINI PRO
def get_model_response(file,query):

    #Spilit the context into managable chunks
    text_spliettr = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)
    context = "\n\n".join(str(p.page_content) for p in file)

    data = text_spliettr.split_text(context)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    searcher = FAISS.from_texts(data, embeddings)


    q = "Which employee has maximum salary?"
    records = searcher.get_relevant_document(q)
    print(records)

    prompt_template = """
        You have to answer the question from the provided context and make sure that you provide all the details\n
        Context: {context}?\n
        Question: {question}\n

        Answer:
    """

    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
    
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.9)
    
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    
    response = chain(
        {
            "input_documents":records,
            "question":query
        },
        return_only_outputs=True
    )
    return response['Output_text']
