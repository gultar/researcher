from src.web_summarize import summarize_web
from src.file_utils import write_file


urls = [
  "https://towardsdatascience.com/develop-your-first-ai-agent-deep-q-learning-375876ee2472?gi=f4a05be1ee20"
]
summaries_str = summarize_web(urls=urls)
write_file("./summaries/q-deep-learning-agent.md", summaries_str)