import re
from PyPDF2 import PdfReader
from colorama import Fore

def log(*args):
    for arg in args:
        print(Fore.GREEN+arg+Fore.WHITE)

def read_file(filepath):
    file = open(filepath, "r")
    return file.read()

def write_file(filepath, content):
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)

def clean_text(input_text: str) -> str:
    # Replace multiple whitespaces with a single space
    cleaned_text = ' '.join(input_text.split())

    # Replace line breaks and tabulations with a single space
    cleaned_text = cleaned_text.replace('\n', ' ').replace('\t', ' ')

    return cleaned_text


def split_text_into_chunks(text, chunk_size=2000, chunk_overlap=200):
    # Split the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # Initialize variables
    current_chunk = ""
    chunks = []

    # Iterate through sentences to form chunks
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    # Add the last chunk
    chunks.append(current_chunk.strip())

    return chunks

def extract_pdf_text(filepath: str):
    text = ""
    pdf_reader = PdfReader(filepath)
    for page in pdf_reader.pages:
            text += page.extract_text() 
            print(page.extract_text())

    return text

def extract_pdf_text_as_pages(filepath: str):
    text = []
    pdf_reader = PdfReader(filepath)
    for page in pdf_reader.pages:
            text.append(page.extract_text() )
            print(page.extract_text())

    return text 