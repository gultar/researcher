from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from .file_utils import split_text_into_chunks, log, read_file
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

class LongTermMemory:
    def __init__(self, save_path: str = "memory", chunk_size: int = 3000, chunk_overlap: int = 200):
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
        except Exception:
            return False

    def load(self, path=""):
        if path == "":
            path = self.save_path
        try:
            self.vectorstore = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"Memory loaded from {path}")
        except Exception as e:
            print(f"Could not load memory from {path}: {e}")
            self.vectorstore = None

    def save(self, path=""):
        if path == "":
            path = self.save_path
        if hasattr(self, "vectorstore") and self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"Memory saved to {path}")

    def get_vectorstore(self, path=""):
        if path == "":
            path = self.save_path
        try:
            return FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            return None

    def add_texts(self, texts: list):
        chunks = []
        for text in texts:
            chunks.extend(
                split_text_into_chunks(text, self.chunk_size, self.chunk_overlap)
            )
        if not chunks:
            return
        if hasattr(self, "vectorstore") and self.vectorstore:
            self.vectorstore.add_texts(chunks)
        else:
            self.vectorstore = FAISS.from_texts(chunks, self.embeddings)

    def add_pdf(self, pdf_path: str):
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.add_texts([text])

    def add_file(self, file_path: str):
        text = read_file(file_path)
        self.add_texts([text])

    def retrieve(self, query: str, top_k: int = 3) -> str:
        if not hasattr(self, "vectorstore") or self.vectorstore is None:
            return ""
        docs = self.vectorstore.similarity_search(query, k=top_k)
        return "\n\n".join(doc.page_content for doc in docs)
