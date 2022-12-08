'''
Takes in the json for a ipynb notebook, extracts the code text and returns it
'''
import json

def json_to_code(filename) -> str:
    '''
    The filename should be a path to the json for an already-run ipynb notebook.
    It should be a list of cells.
    Each cell is a dictionary with "cell_type", "metadata", and "source" as keys.
    We want to grab the source values of cells with cell_type == "code" and return them all as one string (with the appropriate newlines).
    '''
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            notebook = json.load(f)
        except json.decoder.JSONDecodeError:
            print(f"Error decoding {filename}")
            raise
    code = ""
    for cell in notebook:
        if cell["cell_type"] == "code":
            code += f'{"".join(cell["source"])}\n\n'
    return code.rstrip()

if __name__ == "__main__":
    code = json_to_code("Even Chapters/sol2.1.json")
    with open("test.txt", 'w', encoding='utf-8') as f:
        f.write(code)