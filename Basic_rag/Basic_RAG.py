import os
from groq import Groq
import time
def load_file():
    with open("C:/Users/nani3/Desktop/deep_learning projects/Basic_rag/Ai.txt",'r') as f:  
        file=f.read()
    return file
def chunking(file,chunk_size=300, overlap=50):
    file=file.lower()
    chunks=[]
    words =file.split()
    step= chunk_size-overlap
    for i in range(0,len(words),step):
        chunk=words[i:i+chunk_size]
        chunk=" ".join(chunk)
        chunks.append(chunk)
    return chunks

def chunk_score(chunk, question):
    chunk_words = set(chunk.lower().split())
    question_words = set(question.lower().split())
    score = len(chunk_words.intersection(question_words))
    return score
    

def relevent_chunks(question,chunks,min_score):
    best_chunk=None
    best_score=-1
    for chunk in chunks:
        score=chunk_score(chunk,question)
        if score>best_score:
            best_score=score
            best_chunk=chunk
    if best_score<min_score:
        return None
    return best_chunk

def model(question,r_chunk):
    time_n= time.time()


    client=Groq(api_key=os.getenv("groq_api_key"))
    responce= client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role":"system",
             "content":"A friendly Ai"},
             {"role":"user","content":f"Context:\n{r_chunk}\n\nQuestion: {question}\nAnswer:"}
        ],
        temperature=0.3,
        max_completion_tokens=512,

    )
    
    res=responce.choices[0].message.content
    stop_t=time.time()
    t= float(stop_t-time_n)
    return res,t
    
if __name__== '__main__':
    print("Loading document...\n")
    file= load_file()
    chunks= chunking(file)
    print("chunking the words.......\..\n")
    question=input("Ask a question........?")
    print("checking the relevent chunks................................\n")
    min_score=2
    r_chunk=relevent_chunks(question,chunks,min_score)
    if r_chunk is None:
        print("Sorry, the answer is not present in the document.")
        exit()
    model_answer,t=model(question,r_chunk)
    print("Ai generated answer is  ------>> \n")
    print(model_answer)
    print(t)

    
    










