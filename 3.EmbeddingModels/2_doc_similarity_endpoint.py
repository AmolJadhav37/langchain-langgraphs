from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

documents = ["virat kohli is a great batsman in fact he is the best in the world",
             "sachin tendulkar is a god of cricket",
             "messi is a great footballer",
             "cricket is a popular sport in india",
             "football is a popular sport in the world"]

query = "who is the best batsman in the world?"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
print(cosine_similarity([query_embedding], doc_embeddings))
