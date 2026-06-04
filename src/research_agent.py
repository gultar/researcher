from dotenv import load_dotenv
import json
from datetime import date
from .file_utils import write_file, clean_text, split_text_into_chunks, extract_pdf_text, log
from .web_search import get_content_from_url, search_ddg
import time
from .agent import Agent, _load_llm_params
from typing import Optional
from .memory import LongTermMemory
from colorama import Fore

load_dotenv()

today = date.today()
today = str(today)

class Researcher(Agent):
    def __init__(
            self,
            role="researcher",
            model=None,
            number_of_sources=12,
            chunk_size=2000,
            chunk_overlap=200,
            params: dict = None,
        ):
        if params is None:
            params = _load_llm_params()

        super().__init__(role=role, model=model, params=params)

        self.initial_instructions = """
            You are a world-class researcher and journalist. Your task is to research topics and write fact-based
            and well written articles for a specialized or general public. You always respect the truth and provide
            accurate information. You always cite your sources.
        """
        self.messages = [{"role": "system", "content": self.initial_instructions}]
        self.number_of_sources = number_of_sources
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.memory = None

    def init_memory(self, save_path="memory", chunk_size=None, chunk_overlap=None):
        self.memory = LongTermMemory(
            save_path=save_path,
            chunk_size=chunk_size or self.chunk_size,
            chunk_overlap=chunk_overlap or self.chunk_overlap,
        )

    def search(self, query: str):
        print(Fore.CYAN + f"Searching for: {query}" + Fore.WHITE)
        results = search_ddg(query, max_results=self.number_of_sources)
        texts = []
        for result in results:
            url = result.get("href") or result.get("url", "")
            if not url:
                continue
            try:
                content = get_content_from_url(url)
                if content:
                    texts.append(clean_text(content))
                    print(Fore.GREEN + f"  Loaded: {url}" + Fore.WHITE)
            except Exception as e:
                print(Fore.RED + f"  Failed: {url} ({e})" + Fore.WHITE)
            time.sleep(0.5)
        return texts

    def load_data(self, texts: list, save_path="memory", chunk_size=None, chunk_overlap=None):
        if self.memory is None:
            self.init_memory(save_path=save_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.memory.add_texts(texts)
        self.memory.save()

    def load_url(self, url: str):
        content = get_content_from_url(url)
        if content and self.memory:
            self.memory.add_texts([clean_text(content)])

    def load_pdf(self, pdf_path: str):
        text = extract_pdf_text(pdf_path)
        if text and self.memory:
            self.memory.add_texts([text])

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        if self.memory is None:
            return ""
        return self.memory.retrieve(query, top_k=top_k)
