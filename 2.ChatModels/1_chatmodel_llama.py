from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant"
)

reslt = model.invoke("Write a 5 line poem about computers")
print(reslt)