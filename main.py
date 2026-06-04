from dotenv import load_dotenv
from colorama import Fore
from src.research_agent import Researcher
from src.file_utils import read_file, extract_pdf_text
from src.md_to_ipynb import convert_markdown_files_to_notebooks
from src.agent import create_client, _load_llm_params
import yaml
import os

load_dotenv()

# ---------------------------------------------------------------------------
# Load parameters
# ---------------------------------------------------------------------------
param_file = read_file("./parameters.yaml")
research_parameters = yaml.safe_load(param_file)

# LLM provider settings
llm_params = _load_llm_params()
llm_provider = llm_params.get("llm_provider", "openai")
llm_model = llm_params.get("llm_model", "gpt-4o")
llm_smaller_model = llm_params.get("llm_smaller_model", "gpt-4o-mini")

# Research settings
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

print(f"LLM Provider: {llm_provider}")
print(f"LLM Model: {llm_model}")
print(f"LLM Smaller Model: {llm_smaller_model}")
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

# ---------------------------------------------------------------------------
# LLM client (provider-agnostic via create_client)
# ---------------------------------------------------------------------------
client = create_client(llm_params)

researcher = Researcher(
    number_of_sources=research_number_of_sources,
    params=llm_params,
)

system_msg = """
You are a helpful assistant whose task is to create detailed researches.
First, a plan for the research will be designed, then you will have to cycle through each section
and create the content for that section.
"""

messages = [
    {
        "role": "system",
        "content": system_msg
    },
]

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def create_directory(directory_path):
    try:
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_path}' already exists.")


def write_file(filepath, content):
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)


# ---------------------------------------------------------------------------
# LLM wrappers
# ---------------------------------------------------------------------------
def llm_output(prompt, use_smaller=False):
    """
    Send a conversational prompt to the configured LLM (OpenAI or Ollama).
    Maintains the global message history for multi-turn context.
    """
    model = llm_smaller_model if use_smaller else llm_model
    messages.append({"role": "user", "content": prompt})
    print(Fore.LIGHTGREEN_EX + prompt + Fore.WHITE)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    print(reply)
    return reply


def llm_single_output(prompt, use_smaller=True, temperature=0.1):
    """
    Send a one-shot prompt to the configured LLM without persisting history.
    """
    model = llm_smaller_model if use_smaller else llm_model
    msgs = [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}]
    print(Fore.LIGHTGREEN_EX + prompt + Fore.WHITE)
    response = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=temperature,
    )
    reply = response.choices[0].message.content
    print(reply)
    return reply


# Keep old names as aliases for backwards compatibility
def gpt4_output(prompt):
    return llm_output(prompt, use_smaller=False)


def gpt_text_output(prompt, save=False, temperature=0.1):
    return llm_single_output(prompt, use_smaller=True, temperature=temperature)


# ---------------------------------------------------------------------------
# Research plan creation
# ---------------------------------------------------------------------------
def create_research_plan_list(context: str):
    plan_creation_message = f"""
I need to come up with a plan for the following research : {research_description}
The research's content needs to be in {research_language}
Make sure you design relevant content for each section.
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

Output ONLY valid YAML. Do not include any other text, explanation or markdown fences.
"""
    plan = llm_single_output(plan_creation_message, use_smaller=False)
    return plan


def create_research_plan():
    researcher.init_memory(
        save_path=f"./research_projects/{research_dirname}/{research_dirname}_memory",
        chunk_size=research_chunk_size,
        chunk_overlap=research_chunk_overlap,
    )

    if researcher.memory.exists():
        researcher.memory.load(
            f"./research_projects/{research_dirname}/{research_dirname}_memory"
        )
    else:
        texts = researcher.search(research_description)
        researcher.load_data(
            texts,
            save_path=f"./research_projects/{research_dirname}/{research_dirname}_memory",
        )

    context = researcher.retrieve_context(research_description, top_k=5)
    plan = create_research_plan_list(context)
    write_file(f"./research_projects/{research_dirname}/{research_filename}-plan.yaml", plan)
    return plan


# ---------------------------------------------------------------------------
# Section / topic generation
# ---------------------------------------------------------------------------
def create_section_introduction(section_data: dict, section_plan_yaml: str):
    section_title = section_data.get("title", "")
    context = researcher.retrieve_context(section_title, top_k=2)
    section_introduction_prompt = f"""
        You are a world-class researcher.
        Create a detailed introduction for this section of the research.
        The content needs to be in {research_language}

        Section plan:
        {section_plan_yaml}

        Contextual information:
        {context}

        Write the section title as a Markdown H2 heading, then write the introduction.
        Do not yet cover the individual topics — those will follow separately.
    """
    introduction_created = llm_single_output(section_introduction_prompt, use_smaller=False)
    return introduction_created


def expand_topic(topic_data: dict, section_plan_yaml: str):
    topic_title = topic_data['topic']
    topic_content = topic_data["description"]
    context = researcher.retrieve_context(topic_content, top_k=2)

    topic_creation_prompt = f"""
        You are a world-class researcher.
        As a reference, this is the plan for this section:
        {section_plan_yaml}
        Please provide the most detailed content possible with as many examples, code examples,
        quotes and formulas as possible.
        Don't write an introduction since it has already been written before.
        Only focus on the topic provided.
        The content needs to be in {research_language}
        Only focus on this topic in-depth, but not the other topics.

        Here is the topic covered:
        {topic_title}
        Description: {topic_content}

        Contextual information:
        {context}

        Always follow this format when creating your content:
        ### Topic title
        <content>
    """
    topic_created = llm_single_output(topic_creation_prompt, use_smaller=False)
    return topic_created


def provide_explanations(content: str):
    explanation_prompt = f"""
        Given the following content, provide additional explanations, examples and analogies
        to make it more accessible and understandable.
        Content:
        {content}
    """
    return llm_single_output(explanation_prompt, use_smaller=False)


def split_section_in_subsections(section: str):
    split_prompt = f"""
        Split the following section into subsections.
        Each subsection should have a clear title and content.
        Output as Markdown.
        Section:
        {section}
    """
    return llm_single_output(split_prompt, use_smaller=True)


def isolate_topics(section: str):
    isolate_prompt = f"""
        Extract the list of topics covered in the following section.
        Output as a YAML list:
        - topic: "topic title"
        Section:
        {section}
    """
    return llm_single_output(isolate_prompt, use_smaller=True)


def create_quizz(content: str):
    quizz_prompt = f"""
        Create a multiple-choice quiz based on the following content.
        Include 5 questions with 4 answer options each, and mark the correct answer.
        Output as Markdown.
        Content:
        {content}
    """
    return llm_single_output(quizz_prompt, use_smaller=True)


def create_quizzes_for_section(section: str):
    return create_quizz(section)


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------
def create_directories():
    create_directory(f"./research_projects/{research_dirname}")
    create_directory(f"./research_projects/{research_dirname}/individual-sections")


def load_research_data():
    if researcher.memory is None:
        researcher.init_memory(
            save_path=f"./research_projects/{research_dirname}/{research_dirname}_memory",
            chunk_size=research_chunk_size,
            chunk_overlap=research_chunk_overlap,
        )

    # Load URLs
    if research_urls_list:
        for url in research_urls_list:
            researcher.load_url(url)

    # Load PDFs
    if research_pdfs_list:
        for pdf_path in research_pdfs_list:
            researcher.load_pdf(pdf_path)

    # Load text files
    if research_text_files:
        for text_file in research_text_files:
            researcher.memory.add_file(text_file)

    # Web search if no preloaded sources
    if not research_urls_list and not research_pdfs_list and not research_text_files:
        texts = researcher.search(research_description)
        researcher.load_data(
            texts,
            save_path=f"./research_projects/{research_dirname}/{research_dirname}_memory",
        )

    researcher.memory.save()


def create_course_exercises(course: str):
    create_directory(f"./research_projects/{research_dirname}/quizzes")
    sections = course.split(separator)
    quizzes = []
    for index, section in enumerate(sections):
        nb = index + 1
        quizz = create_quizzes_for_section(section)
        quizzes.append(quizz)
        write_file(
            f"./research_projects/{research_dirname}/quizzes/{str(nb)}-{research_filename}-quizz.md",
            section + "\n\n" + quizz,
        )


def generate_research_content(plan: str):
    research = []
    section_dict = yaml.safe_load(plan)

    for section_data in section_dict:
        section_plan_yaml = yaml.dump([section_data], default_flow_style=False)
        section_created = []

        introduction_created = create_section_introduction(
            section_data=section_data, section_plan_yaml=section_plan_yaml
        )
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
        nb = index + 1
        write_file(
            f"./research_projects/{research_dirname}/individual-sections/{research_filename}-section-{str(nb)}.md",
            section,
        )

    if research_convert_to_notebook:
        create_directory(f"./research_projects/{research_dirname}/notebooks")
        convert_markdown_files_to_notebooks(
            f"./research_projects/{research_dirname}/individual-sections/",
            f"./research_projects/{research_dirname}/notebooks",
        )

    return research_content


def generate_exercices_for_course():
    if researcher.memory is None:
        researcher.init_memory(
            save_path=f"./research_projects/{research_dirname}/{research_dirname}_memory",
            chunk_size=research_chunk_size,
            chunk_overlap=research_chunk_overlap,
        )
        researcher.memory.load(
            f"./research_projects/{research_dirname}/{research_dirname}_memory"
        )
    course = read_file(f"./research_projects/{research_dirname}/{research_filename}.md")
    create_course_exercises(course)
    create_directory(f"./research_projects/{research_dirname}/quizzes")
    create_directory(f"./research_projects/{research_dirname}/quizzes-notebooks")
    convert_markdown_files_to_notebooks(
        f"./research_projects/{research_dirname}/quizzes",
        f"./research_projects/{research_dirname}/quizzes-notebooks",
    )


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
