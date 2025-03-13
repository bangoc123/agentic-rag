# Save data to Chroma DB
import os
import pandas as pd
from flask import Flask, request, jsonify, Blueprint
from werkzeug.utils import secure_filename
from openai import OpenAI
from dotenv import load_dotenv
import re
import chromadb
import uuid

chroma_client = chromadb.PersistentClient("db")

load_dotenv()

# Flask App Initialization
app = Flask(__name__)


# OpenAI Client for embeddings
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Use correct OpenAI API key

training_bp = Blueprint('training', __name__, url_prefix='/training')

def get_embedding(text: str) -> list[float]:
    """Generates an embedding for a given text using OpenAI."""
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def sanitize_collection_name(name: str) -> str:
    """Sanitize collection name to be MongoDB-compatible."""
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)  # Replace invalid characters with '_'
    name = name.strip("_")  # Remove leading/trailing underscores
    return name.lower()  # Convert to lowercase for consistency

def sanitize_metadata(record):
    """Convert None values in metadata to empty string or default values, and exclude embeddings."""
    sanitized_record = {
        k: (str(v) if v is not None else "") for k, v in record.items() if k != "embedding"
    }
    return sanitized_record

df = pd.read_csv("./hoanghamobile.csv")


import ast
def join_string(item):
    for i in range(len(item)):
        title, product_promotion, product_specs, current_price, color_options = item

        final_string = ""
        if title:
            final_string += f"{title}"

        if product_promotion:
            product_promotion = product_promotion.replace("<br>", " ").replace("\n", " ")
            final_string += f" {product_promotion}"

        if product_specs:
            product_specs = product_specs.replace("<br>", " ").replace("\n", " ")
            final_string += f" {product_specs}"

        if current_price:
            final_string += f" có giá: {current_price}"

        if color_options:
            final_string += " có màu sắc: "
            colors = ast.literal_eval(color_options)

            final_string += ", ".join(colors)


    return final_string

df['information'] = df[
    [
     'title',
     'product_promotion',
     'product_specs',
     'current_price',
     'color_options']
    ].astype(str).apply(join_string, axis=1)

# Display the DataFrame to confirm
print(df.head())

df = df.head(2) 

# Prepare data
df = df[df['information'].notna()]
df["embedding"] = df["information"].apply(get_embedding)

# Metadata
metadatas = [{"information": row["information"]} for _, row in df.iterrows()]
ids = [str(uuid.uuid4()) for _ in range(len(df))]

# ChromaDB setup
chroma_client = chromadb.PersistentClient("db")
collection_name = "products"
collection = chroma_client.get_or_create_collection(name=collection_name)

# Insert data
collection.add(
    ids=ids,
    embeddings=df["embedding"].tolist(),
    metadatas=metadatas
)

print(f"Inserted {len(df)} documents into ChromaDB collection '{collection.name}'.")
