import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
import os
import re

def convert_markdown_files_to_notebooks(directory_path, output_dir):
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            markdown_file_path = os.path.join(directory_path, filename)
            notebook = convert_markdown_to_notebook(markdown_file_path)
            # print(notebook)
            output_notebook_file = filename.replace('.md', '.ipynb')
            
            with open(f"{output_dir}/{output_notebook_file}", 'w', encoding='utf-8') as f:
                nbformat.write(notebook, f)


def convert_markdown_to_notebook(markdown_file):
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()

    notebook = new_notebook()

    # Use regex to identify code blocks and their languages
    pattern = re.compile(r'```([a-zA-Z]+)\s*([\s\S]+?)```')
    matches = pattern.finditer(content)

    last_end = 0

    for match in matches:
        start, end = match.span()
        language, block = match.groups()

        # Add any text between the last match and the current match as a Markdown cell
        markdown_text = content[last_end:start]
        if markdown_text.strip():
            notebook.cells.append(new_markdown_cell(markdown_text))

        # Code cell with metadata specifying the language
        code_cell = new_code_cell(block)
        code_cell['metadata']['language'] = language.lower()
        notebook.cells.append(code_cell)

        last_end = end

    # Add any remaining text after the last match as a Markdown cell
    remaining_text = content[last_end:]
    if remaining_text.strip():
        notebook.cells.append(new_markdown_cell(remaining_text))

    return notebook



