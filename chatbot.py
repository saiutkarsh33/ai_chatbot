import pytesseract
from langchain import ConversationChain
from langchain.llms import OpenAI
from langchain import VectorDBQA
from langchain.document_loaders import  OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
import pinecone

OPENAI_API_KEY= ""

PINECONE_API_KEY = ''
PINECONE_API_ENV = ''
index_name = "nika-chatbot"

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

loader = OnlinePDFLoader("https://verra.org/wp-content/uploads/2023/11/VM0007-REDD-Methodology-Framework-v1.7.pdf")

data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)


pinecone.init(
    api_key=PINECONE_API_KEY, 
    environment=PINECONE_API_ENV
)


docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)


llm = OpenAI(
    temperature=0.7, 
    openai_api_key=OPENAI_API_KEY)

qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch, return_source_documents=True)

conversation = ConversationChain(llm = llm)
while (True):
    query = input("Ask me: ")
    if (query.lower()=="quit"):
        break
    print(f"\nResponse: {conversation.predict(input=query)}\n\n")
    
