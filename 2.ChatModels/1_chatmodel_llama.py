from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",temperature=1.5
)

reslt = model.invoke("suggest me five indian names for a boy considering new generation don't write any description of the names just write the names")
print(reslt.content)