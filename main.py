import glob
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from fastapi import FastAPI, UploadFile
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from fastapi import File, UploadFile
from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain import PromptTemplate, HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.docstore.document import Document
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List
import os
import re
import uuid
import logging
import warnings
import datetime
import boto3

#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

base_dir=os.getcwd()
###################
app = FastAPI()

aws_access_key_id = "Paste your aws_access_key_id here"
aws_secret_access_key = "Paste your aws_secret_access_key here"
bucket_name = "Paste your bucket_name here"  # Replace with your S3 bucket name
region_name = "Paste your region_name here"  # Replace with your AWS region

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
if not os.path.exists(os.path.join(base_dir, "pdf_folder")):
    os.mkdir(os.path.join(base_dir, "pdf_folder"))

def text_processing(pdf_key):
    # Initialize AWS Textract client
    textract_client = boto3.client('textract', region_name=region_name,
                                   aws_access_key_id=aws_access_key_id,
                                   aws_secret_access_key=aws_secret_access_key)
    # Read the PDF using AWS Textract directly from S3
    response = textract_client.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': pdf_key[0]}})
    # Get the JobId from the response
    job_id = response['JobId']
    print(f'Textract JobId: {job_id}')
    # Wait for the job to complete
    while True:
        response = textract_client.get_document_text_detection(JobId=job_id)
        status = response['JobStatus']
        if status in ['SUCCEEDED', 'FAILED']:
            break
    # Check if the job succeeded
    if status == 'SUCCEEDED':
        # Extracted text content
        doc = []
        selected_pdf = set()
        # Extract text content from Textract response
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                print(item['Text'])
                doc.append(Document(page_content=item['Text'], metadata={"source": pdf_key, "page": item['Page']}))
                selected_pdf.add(pdf_key.split('||'))
        return doc, list(selected_pdf)[0]
    else:
        print(f'Textract job failed with status: {status}')
        return [], ""


def creating_vector_embedding(chunks, collection):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = SentenceTransformerEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)
    persist_directory = 'DB'
    # id =str(uuid.uuid4())
    collection =collection.split('||')[0]
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings,
                                     persist_directory=persist_directory, collection_name=collection)
    vectordb.persist()
    print(id)


@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    if not os.path.exists(os.path.join(base_dir,"temp")):
        os.mkdir(os.path.join(base_dir,"temp"))
    if not os.path.exists(os.path.join(base_dir,"pdf_folder")):
        os.mkdir(os.path.join(base_dir,"pdf_folder"))
    for i in os.listdir(os.path.join(base_dir,"pdf_folder")):
        if file.filename in i:
            result = f"{file.filename} already exists "
            break
    else:
        id = str(uuid.uuid4())
        file_path = os.path.join(base_dir,"temp", f'{id}||{file.filename}')
        file_ = os.path.join(base_dir, "pdf_folder", f'{id}||{file.filename}')
        with open(file_path, "wb") as file_object:
            file_object.write(file.file.read())
        with open(file_, "wb") as file_object:
            file_object.write(file.file.read())
        path = glob.glob(os.path.join(base_dir,"temp", f'{id}||{file.filename}'))
        chunks, collection_name =text_processing(path)
        creating_vector_embedding(chunks, collection_name)
        result = f'embedding is creating {file.filename}'
    return {'text': result}

@app.get("/upload_file_list")
async def upload_file_list():
    path_list = os.listdir(os.path.join(base_dir,'pdf_folder'))
    return {'list': path_list}

@app.post('/delete_particular_file')
async def delete_file(pdf_str:List[str]):
    pdf_name =pdf_str[0]
    collection =pdf_name.split('||')[0]
    persist_directory = 'DB'
    model_name = "sentence-transformers/all-mpnet-base-v2"
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = SentenceTransformerEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)
    # load in
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings,collection_name=collection)
    collection = vectordb._collection
    print(collection,'uuuuuuuuuuuuuuuuuuuuuuuu')
    if pdf_name in os.listdir(os.path.join(base_dir,'pdf_folder')):
        os.remove(os.path.join(base_dir,'pdf_folder',pdf_name))
        deleted = collection.delete(where={"source": pdf_name})
        res = f'file is deleted {pdf_name}'
        collection = vectordb._collection
        print(collection,'vvvvvvvvvvvvvvvvv')
    else:
        res = f'file is not present {pdf_name}'

    return {'status':res}

class ques(BaseModel):
    query: str
    pdf_name: str

@app.post('/question_answer')
async def ques_ans(item :ques):
    logging.basicConfig(filename='LogFile2.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    log = logging.getLogger()
    parameter = item.dict()
    query =parameter['query']
    collections = parameter['pdf_name'].split('||')[0]
    print(collections,'nnnnnnnnnnnnnnnnnnnnnnnnnnnnn')
    persist_directory = 'DB'
    prompt_template = """
            Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

            page_content: {context}

            Question: from mentioned context, {question}

            Answer :

            Provide the most specific and correct answer from the most relevant in source_documents.

            If the answer is not found in the context, strictly respond with this line only - "context does not provide an answer to your question".

            If the answer is not found in the context, You don't try to make up an answer..
            """

    model_name = "sentence-transformers/all-mpnet-base-v2"
    encode_kwargs = {'normalize_embeddings': False}
    embedding = SentenceTransformerEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)

    # load in
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding,
                      collection_name=collections)
    retriever = vectordb.as_retriever(search_type="similarity")  # search_kwargs={"k": 8}
    llm = HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                          model_kwargs={"temperature": 0, "max_length": 100},
                          huggingfacehub_api_hf='Paste your hugging_face token(write) here',
                          )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    question_answers = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=False,
        return_source_documents=True,
        chain_type_kwargs={
            "verbose": False,
            "prompt": prompt,
        }
    )
    response = question_answers({"query": query})
    print(response)
    log.info(f"\n{datetime.datetime.now()} || {query} || {response['result']}\n")
    return response['result']    

if __name__ == "__main__":
    uvicorn.run(app, host= '0.0.0.0', port=8000)
