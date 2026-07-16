from openai import OpenAi
import os
from sentence_transformers import SentenceTransformer
import pdfplumber
import faiss
import groq
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()
emb_model=SentenceTransformer()


def get_r(resume):
    text=""
    with pdfplumber.open(resume) as pdf:
        for page in pdf.pages:
            pdf_text=page.extract_text()
            if pdf_text:
                text+=pdf_text+"\n"
    return text.strip()


def get_responce(resume,indices):
    client=OpenAi(base_url = "https://integrate.api.nvidia.com/v1",
                  api_key=os.getenv("nvidia_api_key")
                )
    responce= client.chat.completion.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role":"system","content":"A Good Analyzer"},{"role":"user","content":indices}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=False
    )
    return responce.choices[0].message


        

if __name__=='__main__':
    print("enter the job description to get match score: \n")
    job_d= input().strip()
    resume="c:/Users/nani3/Desktop/cv/vuyyalawada kiran cv.pdf"
    resume=get_r(resume)
    resume=emb_model.encode(resume)
    description= emb_model.encode(job_d)

    chunks=[chunk for chunk in resume if chunk]
    embedding=emb_model.encode(chunks)
    embedding= embedding.astype("float32")
    index=faiss.IndexFlatL2(embedding.shape[1])
    index.add(embedding)
    indices,distance=index.search(np.array(prompt),5)
    responce=get_responce(resume,indices)
    





