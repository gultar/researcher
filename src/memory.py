from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from .file_utils import split_text_into_chunks, log, read_file
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()

class LongTermMemory:
    def __init__(self, save_path: str ="memory", chunk_size:int = 3000, chunk_overlap:int = 200):
        self.data = []
        self.file_paths = []
        self.texts_array = []
        self.save_path = save_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        

    def exists(self):
        try:
            vectorstore = self.get_vectorstore(self.save_path)
            if vectorstore:
                return True
            else:
                return False
        except:
            return False
        
    def load(self, path=""):
        if path == "":
            path = self.save_path
        self.vectorstore = self.get_vectorstore(path)
        return self.vectorstore

    def get_text_chunks(self, raw_text):
        chunks = split_text_into_chunks(raw_text, self.chunk_size, self.chunk_overlap)
        return chunks
    
    @staticmethod
    def get_pdf_text(pdf_path):
        print(pdf_path)
        text = ""
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        return text
    
    def store(self, file_paths: list[str]= [], texts_array: list[str] = []):
        memory_exists = self.exists()
        if memory_exists:
            log(f"Storing {len(texts_array)} chunks in existing memory at {self.save_path}")
            self.add(file_paths=file_paths, texts_array=texts_array)
        else:
            log(f"Creating new memory at {self.save_path} using {len(texts_array)} texts")
            self.new_vectorstore(file_paths=file_paths, texts_array=texts_array)          


    def new_vectorstore(self, file_paths: list[str]= [], texts_array: list[str] = []):
        self.file_paths = file_paths
        self.texts_array = texts_array
        text_chunks = []
        if len(self.file_paths) != 0:
            for path in self.file_paths:
                text = self.get_pdf_text(path)
                chunks = self.get_text_chunks(text)
                text_chunks.extend(chunks)
        
        if len(self.texts_array) != 0:
            for text in self.texts_array:
                chunks = self.get_text_chunks(text)
                text_chunks.extend(chunks)

        self.text_chunks = text_chunks
        self.vectorstore = FAISS.from_texts(texts=self.text_chunks, embedding=self.embeddings)
        self.vectorstore.save_local(self.save_path)
        log(f"Memory {self.save_path} created")

    def get_vectorstore(self, save_path: str):
        self.vectorstore = FAISS.load_local(save_path, self.embeddings)
        return self.vectorstore

    def retrieve(self, query: str, top_k: int=1)-> list:
        docs = self.vectorstore.similarity_search(query)
        top_k_docs = docs[:top_k]

        texts = []
        for doc in top_k_docs:
            texts.append(doc.page_content)

        return texts
    
    def add(self, file_paths: list[str]= [], texts_array: list[str] = []):
        text_chunks = []
        if len(file_paths) != 0:
            for path in file_paths:
                text = self.get_pdf_text(path)
                chunks = self.get_text_chunks(text)
                text_chunks.extend(chunks)
        
        if len(texts_array) != 0:
            for text in texts_array:
                chunks = self.get_text_chunks(text)
                text_chunks.extend(chunks)

        new_vectorstore = FAISS.from_texts(texts=text_chunks, embedding=self.embeddings)
        self.vectorstore.merge_from(new_vectorstore)
        self.vectorstore.save_local(self.save_path)
        log(f"Memory {self.save_path} merged {len(text_chunks)} text chunks")

        return True
    
    

#######################
# example - merging
#######################
# memory = LongTermMemory("coding")
# file = read_file("./openai.md")
# memory.new_vectorstore([],[file])
# content = memory.retrieve("client.chat.completions.create")
# print('Content:', content)

# new_file = read_file("./langchain.md")

# memory.add([],[new_file])
# new_content = memory.retrieve("PromptTemplates are cool things")
# print("New Content", new_content)

