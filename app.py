import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os 
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
load_dotenv()


# urls = ["https://www.youtube.com/watch?v=82QusHUE9jk","https://www.youtube.com/watch?v=rU63Mpr5-Q4","https://www.youtube.com/watch?v=EtsbdwVE_1s","https://www.youtube.com/watch?v=iQJ7EPxvTLY","https://www.youtube.com/watch?v=O8K2UEZzhBo"]

# combined_content = []
# for url in urls:
#     loader = YoutubeLoader.from_youtube_url(f"{url}", add_video_info=False)
#     x = loader.load()
#     for item in x :
#         combined_content.append(item)

# combined_content
# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 500)
# final_documents = text_splitter.split_documents(combined_content)


embeddings = HuggingFaceEmbeddings(model_name = "hkunlp/instructor-large")

# db = FAISS.from_documents(final_documents,embeddings)


new_db = FAISS.load_local(r"D:\All_data_science_project\Langchain\Youtube_transcript_rag\Notebook\faiss_index",embeddings,allow_dangerous_deserialization=True)







groq_api_key = os.getenv('GROQ_API_KEY')
# Define the prompt template
prompt = ChatPromptTemplate.from_template(
"""
You are a helpful assistant. Answer the following Question based on the context:
<context>
{context}
</context>

And only show the translated output in {language}.
"""
)

llm = ChatGroq(model='Gemma2-9b-It', groq_api_key= groq_api_key,temperature=0.7)

# Define the output parser
parser = StrOutputParser()

# Create the document chain
document_chain = create_stuff_documents_chain(llm, prompt, output_parser=parser)

# Create a retriever from the vector store
retriever = new_db.as_retriever()

# Create a retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Define the input query
def translation_into(query, language):
    response = retrieval_chain.invoke({'input': query, 'language': language})
    return response['answer']

# Streamlit app
st.title("Language Translation with Retrieval Chain")

query = st.text_input("Enter your question:")
language = st.text_input("Enter the language for the answer:")

if st.button("Translate"):
    if query and language:
        answer = translation_into(query, language)
        st.write(f"Translated Answer: {answer}")
    else:
        st.write("Please enter both a question and a language.")
