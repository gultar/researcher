from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import date
from .file_utils import write_file, clean_text, split_text_into_chunks, extract_pdf_text, log
from .web_search import get_content_from_url, search_ddg
import time
from .agent import Agent
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
            model="gpt-3.5-turbo-1106",
            number_of_sources = 12,
            chunk_size = 2000,
            chunk_overlap = 200
        ):

        super().__init__(role, model)  # Call the parent class constructor

        self.initial_instructions = """
            You have a worldclass researcher and journalist. Your task is to research topics and write fact-based
            and well written articles for a specialized or general public. You always respect the guidelines and rules
            that are given to you."""
    
        self.number_of_sources = number_of_sources
        self.query = ""
        self.summaries = []
        self.raw_sources = []
        self.memory_save_path = ""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    @staticmethod
    def extract_text_from_pdfs(filepaths = []):
        texts = []
        for path in filepaths:
            text = extract_pdf_text(path)
            log(f"Extracted text from {path}")
            texts.append(text)
            
        return texts
    
    def init_memory(self, save_path="memory", chunk_size: int = 4000, chunk_overlap: int = 200):
        self.memory = LongTermMemory(
            save_path=save_path, 
            chunk_overlap=chunk_overlap, 
            chunk_size=chunk_size,
        )
        
    def gather_data_from_pdfs(self, pdf_paths: list[str] = []):
        texts = self.extract_text_from_pdfs(pdf_paths)
        self.raw_sources = texts
        self.memory.store(texts_array=self.raw_sources)

        return True
    
    def summarize_sources(self, sources: list[str]): #, chunk_size = 2000, chunk_overlap=200
        summaries = []

        for source in sources:
            summary = []
            chunks = split_text_into_chunks(source, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            for chunk in chunks:
                chunk_summary = self.summarize_text(chunk)
                summary.append(chunk_summary)

            summary = "\n\n".join(summary)
            summaries.append(summary)

        self.summaries = summaries
        return summaries

    def gather_data(self, web_query: str = "", urls: list[str] = [], texts: list[str] = [], summarize: bool = False):
        urls_list = []
        if web_query == '' and len(urls) == 0 and len(texts) == 0:
            raise "You need to provide at least one source of data: a web query, a list of urls or a list of texts"
        
        if web_query != '':
            self.query = web_query
            urls_list = self.find_best_articles(web_query)
            log(f"Selected those sources: {urls_list}")

        urls_list.extend(urls)
        documents = get_content_from_url(urls_list)
        self.raw_sources.extend(documents)
        self.raw_sources.extend(texts)

        if summarize: 
            summaries = self.summarize_sources(self.raw_sources)
            summaries = "\n\n".join(summaries)
            print(summaries)
            self.raw_sources.extend(summaries)
            
        self.memory.store(texts_array=self.raw_sources)

        return True

    def research_web_with_memory(self, query: str):
        self.gather_data(query)

        context = self.memory.retrieve(self.query, top_k=1)
        article = self.produce_article(context)
        
        return article

    def research_from_web_search(self, query: str):
        self.query = query
        urls_list = self.find_best_articles(query)
        documents = get_content_from_url(urls_list)
        self.raw_sources.extend(documents)
        summaries = self.summarize_documents(documents)


        self.summaries = summaries
        
        summaries_str = "\n\n".join(summary for summary in summaries if summary is not None)
        self.summaries = summaries_str

        context = self.memory.retrieve(self.query)
        article = self.produce_article(context)
        
        return article
    
    def research_from_url_list(self, query: str, urls_list: list):
        self.query = query
        
        documents = get_content_from_url(urls_list)
        summaries = self.summarize_documents(documents)
        self.raw_sources.extend(documents)

        timestamp = str(time.time() * 1000)
        summaries_str = "\n\n".join(summary for summary in summaries if summary is not None)
        write_file(f"./summaries/{today}-{timestamp}.md", summaries_str)

        content_plan = "" #self.create_content_plan

        article = self.produce_article(summaries)
        
        return article


    def complete(self, prompt: str, temperature: Optional[int]=0):
        return super().complete(prompt, temperature)
    
    def cheaper_complete(self, prompt: str, temperature: Optional[int]=0):
        return super().cheaper_complete(prompt, temperature)
    
    def format_search_results(self, results: list[dict]):
        results_string = ""
        for result in results:
            entry = f'Title: {result["title"]}\nURL: {result["href"]}\nDescription: {result["description"]}\n\n'
            results_string += entry + "\n"

        return results_string
    
    def find_best_articles(self, query: str)-> list[str]:
        results_array = search_ddg(query)
        results_string = self.format_search_results(results_array)
        urls = ""

        prompt = f"""
            You are a world-class researcher whose mission is to find the best and most relevant articles on a topic.
            Here is the data you need to investigate:
            {results_string}
            Listed above are the results for the following search query: {self.query}
            -Please choose the top {str(self.number_of_sources)} most relevant articles relative to the query and only provide results as an array of their URLs/Href. 
            -Do not include anything else.
            -Never include the same article more than once.
            -Avoid including two URLs that are two similar. 
            -Only include one from each source, ideally.
            -Don't topic sources that are from Youtube.com, since you won't be able to consult them.
            -Don't pick URLs that end in .pdf, as you won't be able to consult them.
            -The output MUST be an array of the chosen URLs as string. 
            -Don't add markdown tags, just output the array string.
            """
        urls = self.complete(prompt)
        urls_list = json.loads(urls)
        return urls_list
    
    def organize_summaries(self, summaries_str: str):
        prompt = """
            You are a world-class researcher whose mission is to organize content for the production of articles.
            You are provide with the summary of various articles, and your task it to group and organize content by theme.

            Here is the data you need to investigate:
            {summaries_str}
            -Extract all of the themes covered in the summary.
            -If two elements cover the same exact topic, join them together as one element 
            -Do not include anything else, no acknowledgement.
            -NEVER add markdown tags, just output as a JSON string
            -Output your response as YAML using the following schema example:
        """
        prompt = prompt.replace("{summaries_str}", summaries_str)
        response = self.complete(prompt)
        
        response = response.replace("```json","").replace("```","")
        print(response)
        return response
    
    def extract_code_from_text_chunk(self, text):
        try:
            
            template = f"""
                As a professional code expert, extract any code examples from the
                provided text and add context details in comment. You will be creating a coding tutorial using these examples.
                You need to adhere to these guidelines:

                -Extract any relevant code examples, and add comments detailing its usage.
                -Add all code as markdown code blocks of the relevant language
                -Provide short explanations outside of the blocks detailing its usage
                -Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.
                -Rely strictly on the provided text, without including external information.
                -Do not modify the code, except for the addition of comments
                -Please format the output using Markdown
                Text to summarize: {text}
            """
            results = self.cheaper_complete(template)
            return results
        except Exception as e:
            print(f"An error occurred while summarizing a chunk: {e}")
            return None



    def summarize_text(self, text):

        try:
            #-Any code or mathematical formula should be added in full and in priority, without change.
            template = f"""
            As a professional summarizer, create a concise and comprehensive summary of the given text,
            -The summary should cover all the key points and main ideas presented in the original text, 
            while also condensing the information into a concise and easy-to-understand format.
            -Please ensure that the summary includes relevant details and examples that support the main ideas, 
            while avoiding any unnecessary information or repetition.
            -The length of the summary should be appropriate for the length and complexity of the original text,
            providing a clear and accurate overview without omitting any important information.
            -Format that text as Markdown with titles and subtitles.
            -Keep all code examples intact, without changes
            -Make sure to keep all of the code snippets you find.
            Text to summarize: {text}
            """
            # template = f"""
            #     As a professional summarizer, create a concise and comprehensive summary of the given text, 
            #     be it an article, post, conversation, or passage, while adhering to these guidelines:

            #     -Craft a summary that is detailed, thorough, in-depth, and complex, while maintaining clarity and conciseness.
            #     -Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.
            #     -Rely strictly on the provided text, without including external information.
            #     -Format the summary in paragraph form for easy understanding.
            #     -Make sure the summary is not too long.
            #     -Please format the output using Markdown
            #     -Organize the summary by topic
            #     Text to summarize: {text}
            # """
            return self.cheaper_complete(template)
        except Exception as e:
            print(f"An error occurred while summarizing a chunk: {e}")
            return None
    
    def summarize_documents(self, documents):

        summaries = []
        # documents = splitter.split_documents(data)
        for document in documents:
            chunks = split_text_into_chunks(document)

            for chunk in chunks:
                
                summary = self.summarize_text(chunk)
                
                summaries.append(summary)
                
        return summaries
    
    def extract_code_from_documents(self, documents):
        summaries = []
        # documents = splitter.split_documents(data)
        for document in documents:
            chunks = split_text_into_chunks(document)

            for chunk in chunks:
                
                summary = self.extract_code_from_text_chunk(chunk)
                
                summaries.append(summary)
                
        return summaries

    def generate_title(self, article: str):
        today = date.today()
        today = str(today)

        template=f"""
                You are a world-class journalist, and you will try to creation an appropriate title for the following text.
                Please follow all of the following rules:
                1. Format the title without special characters. Only use letters, numbers and hyphens (-) to separate words.
                2. Make it a valid filename string on any operating system
                3. Only output the filename, nothing else. No comment, no acknowledgement, no explanations. Just the filename.
                Here is a list of examples to give you an idea of the format:
                    blog-post-example
                    example-blog-post
                    article-blog-post-example
                    post-about-subject
                    blog-post
                TEXT:{article}
        """
        title = self.cheaper_complete(template)

        return f"{today}-{title}"
    
    def produce_article(self, summaries: list, additional_context: str = ""):
        
        conclusions = "\n\n".join(summary for summary in summaries if summary is not None)
        conclusions = clean_text(conclusions)
        template=f"""
            {conclusions}
                You are a world-class journalist, and you will write a blog post based on the following summary.
                {additional_context}
                Please follow all of the following rules:
                - Make sure the content is engaging, informative with good data
                - Make sure the content is detailed, with examples or quotes, or original content
                - The content should address the following topic very well: {self.query}
                - The content needs to be detailed, and very well sourced
                - The content needs to be written in a way that is easy to read and understand
                - Please format the output using Markdown
                - Always put a title, and add sub-titles when necessary, in markdown syntax
                - Write a timestamp under the title with today's date in the following format **Published on YYYY-MM-DD**: {today} 
                SUMMARY:
        """
                
        article = self.complete(template, temperature=0.7)
        return article
    
    def provide_summary_of_sources(self, topic: str, urls_list: list[str] =[]):
        documents = []
        urls_list = self.find_best_articles(topic)
        documents = get_content_from_url(urls_list)
        self.raw_sources.extend(documents)
        summaries = self.summarize_documents(documents)
        return summaries
    
    
    def extract_code_examples(self, topic: str, urls_list: list[str] =[], source_texts: list[str] =[]):
        documents = []
        if len(urls_list) == 0:
            urls_list = self.find_best_articles(topic)

        if len(source_texts) == 0 and len(urls_list) != 0:
            documents = get_content_from_url(urls_list)
        else:
            documents = source_texts
        summaries = self.extract_code_from_documents(documents)
        return summaries
