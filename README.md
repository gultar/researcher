# Research Project Script README

## Overview

This script is designed to automate the process of conducting research and generating content, including sections, topics, quizzes, and exercises. It leverages the OpenAI GPT-4 model for generating content and organizes the output into a structured format suitable for academic or professional research projects.

## Key Features

- **Automated Research Plan Creation:** Generates a detailed research plan based on given parameters.
- **Content Generation:** Utilizes GPT-4 or GPT-3.5 for generating rich and detailed content for each section of the research.
- **Quiz and Exercise Generation:** Automatically creates quizzes and exercises relevant to the research topics.
- **PDF, Text, and Web Data Integration:** Supports loading and integrating content from PDFs, text files, and web sources.
- **Markdown and Jupyter Notebook Conversion:** Converts generated content into Markdown format and Jupyter Notebooks for easy distribution and presentation.

## Dependencies

- `openai`: For accessing the OpenAI GPT-4 model.
- `dotenv`: To manage environment variables.
- `colorama`: For colored terminal output.
- `yaml`: For handling YAML files.
- `os`: For directory and file operations.
- `playwright`: For fetching Website content or results from DuckDuckGo.
- `langchain`: For using various Text Embedding models and Vectorstores ('all-MiniLM-L6-v2' and FAISS in this case).
- Custom modules: `research_agent`, `file_utils`, `md_to_ipynb`.

## Setup

1. **Environment Variables:** Ensure `.env` file is present with necessary API keys and settings.
2. **Install Dependencies:** Run `pip install openai dotenv colorama pyyaml`.

## Usage

1. **Configuration:** Set up the `parameters.yaml` file with your research parameters (language, directories, filenames, etc.).
2. **Running the Script:** Execute the script using `python main.py`.
3. **Automated Processes:** The script will automatically:
   - Create directories for storing research projects.
   - Load research data from various sources.
   - Generate a research plan.
   - Create detailed research content.
   - Generate quizzes and exercises.
   - Convert contents to Markdown and Jupyter Notebooks.

## Functions Overview

- `create_directory`: Creates a new directory.
- `read_file`: Reads content from a file.
- `write_file`: Writes content to a file.
- `gpt4_output`, `gpt_text_output`: Functions for generating text using GPT-4 or GPT-3.5.
- `create_research_plan_list`, `extend_plan`: Functions for creating and extending the research plan.
- `iterate_sections`, `create_section`, `create_section_introduction`, `expand_topic`: Functions for processing and generating content for each section and topic.
- `provide_explanations`, `split_section_in_subsections`, `isolate_topics`, `create_quizz`, `create_quizzes_for_section`: Functions for generating additional content and quizzes.
- `load_research_data`, `create_research_plan`, `generate_research_content`, `generate_exercices_for_course`: High-level functions for executing the research project workflow.
- `main`: The main function to run the entire script.

## Note

This script requires a proper setup of the OpenAI API and an understanding of YAML and Markdown formats. Ensure all custom modules (`research_agent`, `file_utils`, `md_to_ipynb`) are correctly implemented and accessible.