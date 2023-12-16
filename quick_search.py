from openai import OpenAI
from dotenv import load_dotenv
from src.research_agent import Researcher
from src.file_utils import write_file

load_dotenv()

client = OpenAI(
  organization='org-89slefGLSVO0BZUSanFQus9v',
)

filename = "knowledge-graph"
query = "A short research on Knowledge Graphs with Langchain"

researcher = Researcher(
    number_of_sources=3
)
researcher.init_memory(
    "./research_projects/quick-search"
)

researcher.gather_data(
    web_query=query
)
context = researcher.memory.retrieve(query=query, top_k=3)
article = researcher.produce_article(context)

print(article)

write_file(f"./summaries/{filename}.md", article)
