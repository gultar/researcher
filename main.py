from openai import OpenAI
from dotenv import load_dotenv
from colorama import Fore
from src.research_agent import Researcher
from src.file_utils import read_file, extract_pdf_text
from src.md_to_ipynb import convert_markdown_files_to_notebooks
import yaml
import os

load_dotenv()

client = OpenAI(
  organization='org-89slefGLSVO0BZUSanFQus9v',
)

param_file = read_file("./parameters.yaml")
research_parameters = yaml.safe_load(param_file)

number_of_sections = 8
# Extract values
research_language = research_parameters.get('research_language')
research_dirname = research_parameters.get('research_dirname')
research_filename = research_parameters.get('research_filename')
research_short_description = research_parameters.get('research_short_description')
research_description = research_parameters.get('research_description')
research_convert_to_notebook = research_parameters.get('convert_to_notebooks')
research_urls_list = research_parameters.get('urls_list')
research_pdfs_list = research_parameters.get('pdfs_list')
research_text_files = research_parameters.get('text_files')
research_number_of_sources = research_parameters.get('number_of_sources')
research_summarize_sources = research_parameters.get('summarize_sources')
research_chunk_size = research_parameters.get('chunk_size')
research_chunk_overlap = research_parameters.get('chunk_overlap')
research_preload_plan = research_parameters.get('preload_plan')
research_preloaded_plan_path = research_parameters.get('preloaded_plan_path')

separator = "\n\n===================================================\n\n"

print("Research Language:", research_language)
print("Research Directory Name:", research_dirname)
print("Research Filename:", research_filename)
print("Research Short Description:", research_short_description)
print("Research Description:")
print(research_description)

print("Convert to Notebooks:", research_convert_to_notebook)
print("URLs List:", research_urls_list)
print("PDFs List:", research_pdfs_list)
print("Text Files:", research_text_files)
print("Number of Sources:", research_number_of_sources)
print("Summarize Sources:", research_summarize_sources)
print("Chunk Size:", research_chunk_size)
print("Chunk Overlap:", research_chunk_overlap)
print("Preload Plan:", research_preload_plan)
print("Preloaded Plan Path:", research_preloaded_plan_path)

researcher = Researcher(
    number_of_sources=research_number_of_sources,
)

system_msg = """
      You are a helpful assistant whose task is to create detailled researches.
      First, a plan for the research will be designed, then you will have to cycle through each section
      and create the content for that section. 
      """

messages = [
  {
      "role": "system", 
      "content": system_msg
  },
]

def create_directory(directory_path):
    try:
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    except FileExistsError:
        # If the directory already exists, print a message
        print(f"Directory '{directory_path}' already exists.")


def read_file(filepath):
    file = open(filepath, "r", encoding="utf-8")
    return file.read()


def write_file(filepath, content):
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)

def gpt4_output(prompt):
    
    messages.append({"role": "user", "content": prompt})

    print(Fore.LIGHTGREEN_EX+prompt+Fore.WHITE)
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages
    )
    reply = response.choices[0].message.content
    
    messages.append({"role": "system", "content": reply})
    print(reply)
    return reply

def gpt_text_output(prompt, save=False, temperature = 0.1):
    msgs = [{"role": "system", "content": system_msg},{"role": "user", "content": prompt}]
    # messages.append()
    print(Fore.LIGHTGREEN_EX+prompt+Fore.WHITE)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",#"gpt-4-1106-preview",#
        messages=msgs,
        temperature=temperature
    )
    reply = response.choices[0].message.content
    
    if save: messages.append({"role": "system", "content": reply})
    print(reply)
    return reply



def create_research_plan_list(context: str):
    plan_creation_message = f"""
I need to come up with a plan for the following research : {research_description}
The research's content needs to be in {research_language}
Make sure you design revelant content for each section.
Based your plan on this contextual information: {context}

This schema shows you how to organize the content:
Output your response as YAML, and reformat the following schema to valid YAML
Always wrap your string values in double quotes, and never add other double quotes within those strings
- section: 1
  title: "section title"
  topics:
    - topic: "topic title"
      description: "Topic Description ..."
    - topic: "topic title"
      description: "Topic Description ..."
    - topic: "topic title"
      description: "Topic Description ..."

- section: 2
  title: "section title"
  topics:
    - topic: "topic title"
      description: "Topic Description ..."
    - topic: "topic title"
      description: "Topic Description ..."
"""
    print("Generating research plan...")
    plan = gpt4_output(plan_creation_message)
    
    if "```yaml" in plan:
        plan = plan.replace("```yaml","")
        plan = plan.replace("```","")
    
    return plan

def extend_plan(plan: str, context: str):
    plan_creation_message = f"""
You are a world-class course content creator.
I have a plan formatted in YAML for a course on the following subject : {research_description}
I need you to subdivide each lesson into more topics, so I can have a more in-depth course.
Make sure the topics are relevant to content for each section.
Don't repeat the same topic twice.
This schema shows you how to organize the content:
Always resuse the plan's format and output your response as parsable YAML.
Always wrap your string values in double quotes, and never add other double quotes within those strings

Plan: {plan}
"""
    print("Generating research plan...")
    plan = gpt4_output(plan_creation_message)
    
    if "```yaml" in plan:
        plan = plan.replace("```yaml","")
        plan = plan.replace("```","")
    
    return plan

# data = yaml.safe_load(yaml_string)
# yaml_section = yaml.dump([section_data], default_flow_style=False)

def iterate_sections(yaml_string, section_func, topic_func):
    data = yaml.safe_load(yaml_string)

    for section_data in data:
        if section_func: section_func(section_data)
        
        topics = section_data['topics']
        for topic_data in topics:
            if topic_func: topic_func(topic_data)

def create_section(section_data: dict):
    
    section_nb = str(section_data['section'])
    section_title = section_data['title']
    topics = section_data["topics"]
    section_content = yaml.dump([topics], default_flow_style=False)
    section_creation_prompt = f"""
        You are a world-class researcher.
        Your task is to create the following research: {research_short_description}
        Create the complete content for section number {section_nb}.
        Please provide the most detailed content possible with as many examples, code examples
        quotes and formulas as possible. 
        
        The content needs to be in {research_language}
        
        Here is the plan for this section:
        Section: {section_title}
        Content: {section_content}
        """
    
    generated_section = gpt_text_output(section_creation_prompt)

    return generated_section

def create_section_introduction(section_data:dict, section_plan_yaml: str):
    section_nb = str(section_data['section'])
    section_title = section_data['title']
    section_topics = section_data["topics"]
    topics = yaml.dump([section_topics], default_flow_style=False)

    context = researcher.memory.retrieve(section_title)

    section_introduction_prompt = f"""
        You are a world-class researcher.
        You will need to create the introduction this the following Section.
        Please, provide a brief introduction and a presentation of the topics covered by the section.
        Don't create any content for the topics, since they will be created separately.
        Only focus on creating the introduction to this section
        As a reference, this is the plan for this section: 
        {section_plan_yaml}
        The content needs to be in {research_language}.

        Section {section_nb} - {section_title}

        Topics: {topics}
        
        Always follow this format when creating your content:
        
        # Section 1 - Title
        Introduction and section description...
        ## Topics Covered
        Mini-table of content...

        Here is some context to inform your response: 
        {context}
        """
    
    introduction_created = gpt_text_output(section_introduction_prompt)
    return introduction_created

def expand_topic(topic_data: dict, section_plan_yaml:str):
    
    topic_title = topic_data['topic']
    topic_content = topic_data["description"]

    context = researcher.memory.retrieve(topic_content, top_k=2)

    topic_creation_prompt = f"""
        You are a world-class researcher.
        As a reference, this is the plan for this section: 
        {section_plan_yaml}
        Please provide the most detailed content possible with as many examples, code examples
        quotes and formulas as possible.
        Don't write an introduction since it has already been written before. 
        Only focus on the topic provided
        The content needs to be in {research_language}
        Only focus on this topic in-depth, but not the other topics.
                
        Here is the topic covered:
        {topic_title}
        Description: {topic_content}
        
        Always follow this format when creating your content:
        
        ## Topic
        Explanations, Code, Formula, etc.
        ## Topic
        Explanations, Code, Formula, etc.
        
        Here is some context to inform your response: 
        {context}
        """
    
    generated_section = gpt_text_output(topic_creation_prompt)

    return generated_section



def provide_explanations(topic_content: dict, section_plan_yaml:str):
    
    topic_creation_prompt = f"""
        You are a world-class researcher.
        As a reference, this is the plan for this section: 
        {section_plan_yaml}
        -Please provide additional explanations on the following section of the section.
        -If you find a code block or a formula, make sure to add comments to explain it in detail.
        -If a code block is incomplete, i.e. only contains a 
         comment saying the code has to be implement, make sure to implement the code yourself
        -If the section is textual, add more relevant context to it, and provide as much explanation as possible.
        -Don't write an introduction or a conclusion since they have already been written before. 
        -Only focus on the topic provided
        -Only focus on this topic in-depth, but not the other topics.
        -Never add any acknowledgement to me
        -The 
                
        Here is the current content:
        {topic_content}
        
        """
    
    generated_section = gpt_text_output(topic_creation_prompt)

    return generated_section

def split_section_in_subsections(section: str)-> list[str]:
    subsections = section.split("\n#")
    new_subsections = []
    for section in subsections:
        if "# Section" in section:
            new_subsections.append(section)
        else:
            new_subsections.append("\n#" + section)
    return new_subsections

def isolate_topics(section)-> list[str]:
    subsections = split_section_in_subsections(section)
    introduction = ""
    conclusion = ""
    topics = []
    

    for sub in subsections:
        if "# Section" in sub:
            introduction += sub
        elif "# Topics" in sub:
            introduction += sub
        elif "Conclusion" in sub:
            conclusion += sub
        else:
            topics.append(sub)

    return topics

def create_quizz(topic: str):
    context = researcher.memory.retrieve(topic)
    prompt = f"""
    You are a world-class content creator.
    Your task is to create tests and quizzes from the content submitted to you.
    If the content is very short, add a number of questions or assignments proportional to the length
    of the material.
    Make sure tha the quizz or exercises cover the topic very well, and are creative.
    Don't cover subjects that are not covered in the topic provided.
    Always write the answers underneath.
    Never create a research plan. Only create the tests and quizzes.
    Don't add any acknowledgement, just output the tests and quizzes.

    Content: {topic}

    Use the following format:
    # Test
    - question 1
    - question 2
    - question 3

    # Answers
    - answer 1
    - answer 2
    - answer 3

    """
# Here is some context to base your quizzes and exercises on: {context}
    response = gpt_text_output(prompt=prompt, temperature=0.6)
    return response

def create_quizzes_for_section(section: str):
    topics = isolate_topics(section)
    
    topics_str = "\n\n".join(topics)
    quizz_content = create_quizz(topics_str)
    return quizz_content



def create_directories():
    create_directory("./research_projects")
    create_directory("./pdfs")
    create_directory("./plans")
    create_directory("./text_files")
    create_directory("./summaries")
    create_directory("./parameters")
    create_directory(f"./research_projects/{research_dirname}")
    create_directory(f"./research_projects/{research_dirname}/individual-sections")

    return True

def load_research_data(web_search= True, urls_list=True, text_list=True):
    texts = []
    web_query = ""
    
    researcher.init_memory(
        save_path=f"./research_projects/{research_dirname}/{research_dirname}_memory",
        chunk_size=research_chunk_size,
        chunk_overlap=research_chunk_overlap
    )

    if web_search:
        web_query = research_short_description

    for pdf_path in research_pdfs_list:
        texts.append(extract_pdf_text(pdf_path))
        print("Loaded PDF file:", pdf_path)

    for text_path in research_text_files:
        text = read_file(text_path)
        texts.append(text)
        print("Loaded text file:", text_path)

    researcher.gather_data(
        web_query=web_query, 
        urls=research_urls_list,
        texts=texts,
        summarize=research_summarize_sources
    )

    return True

def create_research_plan():
    
    context = researcher.memory.retrieve(query=research_short_description, top_k=4)
    if researcher.summaries and len(researcher.summaries) > 0:
        context.extend(researcher.summaries)
    plan = create_research_plan_list(context=context)
    plan = extend_plan(plan=plan, context=context)
    
    write_file(f"./research_projects/{research_dirname}/plan-{research_filename}.yaml", plan)
    return plan

def create_course_exercises(course: str):
    create_directory(f"./research_projects/{research_dirname}/quizzes")
    sections = course.split(separator)
    quizzes = []
    for index,section in enumerate(sections):
        nb = index + 1
        quizz = create_quizzes_for_section(section)
        quizzes.append(quizz)
        write_file(f"./research_projects/{research_dirname}/quizzes/{str(nb)}-{research_filename}-quizz.md", section+"\n\n"+quizz)


def generate_research_content(plan: str):
    research = []

    section_dict = yaml.safe_load(plan)
    for section_data in section_dict:
        section_plan_yaml = yaml.dump([section_data], default_flow_style=False)
        section_created = []
        
        introduction_created = create_section_introduction(section_data=section_data, section_plan_yaml=section_plan_yaml)
        section_created.append(introduction_created)

        topics = section_data["topics"]
        for topic in topics:
            topic_created = expand_topic(topic_data=topic, section_plan_yaml=section_plan_yaml)
            section_created.append(topic_created)

        section = "\n\n".join(section_created)
        research.append(section)

    research_content = f"{separator}".join(research)
    write_file(f"./research_projects/{research_dirname}/{research_filename}.md", research_content)
    sections = research_content.split(separator)
    for index, section in enumerate(sections):
        nb = index+1
        write_file(f"./research_projects/{research_dirname}/individual-sections/{research_filename}-section-{str(nb)}.md", section)

    if research_convert_to_notebook:
        create_directory(f"./research_projects/{research_dirname}/notebooks")
        convert_markdown_files_to_notebooks(f"./research_projects/{research_dirname}/individual-sections/",f"./research_projects/{research_dirname}/notebooks")

    return research_content


def generate_exercices_for_course():
    researcher.init_memory(
        save_path=f"./research_projects/{research_dirname}/{research_dirname}_memory",
        chunk_size=research_chunk_size,
        chunk_overlap=research_chunk_overlap
    )
    researcher.memory.load(f"./research_projects/{research_dirname}/{research_dirname}_memory")
    course = read_file(f"./research_projects/{research_dirname}/{research_filename}.md")
    create_course_exercises(course)
    create_directory(f"./research_projects/{research_dirname}/quizzes")
    create_directory(f"./research_projects/{research_dirname}/quizzes-notebooks")
    convert_markdown_files_to_notebooks(f"./research_projects/{research_dirname}/quizzes",f"./research_projects/{research_dirname}/quizzes-notebooks")
    

def main():
    create_directories()
    load_research_data()

    if research_preload_plan:
        plan = read_file(research_preloaded_plan_path)
    else:
        plan = create_research_plan()

    generate_research_content(plan)



if __name__ == "__main__":
    main()
