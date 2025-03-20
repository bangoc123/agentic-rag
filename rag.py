import pymongo
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import requests
import os
import numpy as np
import chromadb
from agents import Agent, Runner, function_tool

chroma_client = chromadb.PersistentClient("db")
collection_name='products'

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    return response.data[0].embedding

@function_tool
def rag(query: str) -> str:
    collection = chroma_client.get_collection(name=collection_name)
    query_embedding = get_embedding(query)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Perform vector search
    search_results = collection.query(
        query_embeddings=query_embedding, 
        n_results=3 
    )

    metadatas = search_results.get('metadatas', [])

    search_result = ""
    i = 0

    for i, metadata_list in enumerate(metadatas):
        if isinstance(metadata_list, list):  # Ensure it's a list
            for metadata in metadata_list:  # Iterate through all dicts in the list
                if isinstance(metadata, dict):
                    combined_text = metadata.get('information', 'No text available').strip()

                    search_result += f"{i}). \n{combined_text}\n\n"
                    i += 1
    return search_result


@function_tool
def shop_information_rag():
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials

    # Define the scope
    scope = ['https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive']

    # Add credentials to the account
    credentials = ServiceAccountCredentials.from_json_keyfile_name('mles-class-12c1216b7303.json', scope)

    # Authorize the clientsheet
    client = gspread.authorize(credentials)

    # Get the sheet (by name or by URL)
    # sheet = client.open('Your Sheet Name').sheet1  # For first sheet
    sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1mOkgLyo1oedOG1nlvoSHpqK9-fTFzE9ysLuKob9TXlg').sheet1

    # Example operations
    # Get all values
    data = sheet.get_all_records()

    return data

