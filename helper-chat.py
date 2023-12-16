#!/usr/bin/env python

from openai import OpenAI
from dotenv import load_dotenv
from colorama import Fore
from src.file_utils import read_file
import yaml
from src.memory import LongTermMemory
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI

param_file = read_file("./parameters.yaml")
research_parameters = yaml.safe_load(param_file)

number_of_sections = 8
# Extract values

research_dirname = research_parameters.get('research_dirname')
research_filename = research_parameters.get('research_filename')
research_short_description = research_parameters.get('research_short_description')
research_description = research_parameters.get('research_description')

memory = LongTermMemory(
    f"./research_projects/{research_dirname}/{research_dirname}_memory",
)

try:
    memory.load()
except:
    print("Could not load Long Term Memory. It probably hasn't been created.")


load_dotenv()


retriever = memory.vectorstore.as_retriever()
# Step 2: Create a retriever tool
tool = create_retriever_tool(
    retriever,
    "faiss_retriever",
    "Retrieves documents using FAISS retrieval.",
)
tools = [tool]

# Step 3: Initialize the ReAct agent
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)


# Start the conversation
print("==================================================")
print("""
[SYSTEM]: You are chatting with your Researcher Assistant. 
Type 'exit' to end the conversation. 
Type '!ok' to submit.
""")
print("==================================================")
def multi_input(prompt):
    print("[USER]: ")
    lines = []
    first_line = True
    while True:
        if not first_line:
            prompt = ""
        else:
            first_line = False

        user_input = input(prompt)
        # 👇️ if user pressed Enter without a value, break out of loop
        if "!ok" in user_input.lower():
            user_input = user_input.replace("!ok","")
            lines.append(user_input + '\n')
            break
        else:
            lines.append(user_input + '\n')

    return "\n".join(lines)


while True:
    user_input = multi_input("")
    # Exit the loop if the user types 'exit'
    if 'exit\n' in user_input.lower():
        print("Exiting chat...")
        break
    
    
    # Generate a response from GPT
    print("[SYSTEM]: GPT is answering...")
    agent_executor(user_input)


