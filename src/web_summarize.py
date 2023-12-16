from PyPDF2 import PdfReader
from dotenv import dotenv_values, load_dotenv
from .web_search import get_content_from_url
from .research_agent import Researcher

def summarize_web(urls):
    researcher = Researcher(
        chunk_size=800, 
        chunk_overlap=50
    )

    def write_file(filepath, content):
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)

    load_dotenv()
    env = dotenv_values(".env")

    texts = get_content_from_url(urls)
    summaries = researcher.summarize_sources(texts)
    separator = "\n\n==================================================\n\n"
    summaries_str = f"{separator}".join(summaries)

    return summaries_str
