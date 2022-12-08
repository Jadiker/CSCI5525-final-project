import json
import openai
import os
from dotenv import load_dotenv # install with `pip install python-dotenv`
from enum import Enum
from typing import List

SECRET_FILE = "secret/secret.env"

# Load the secret file
load_dotenv(SECRET_FILE)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLD_FINETUNED_MODEL = os.getenv("FINETUNED_MODEL")
NEW_FINETUNED_MODEL = os.getenv("NEW_FINETUNED_MODEL")
openai.api_key = OPENAI_API_KEY

# the examples you want completions for
# all test solutions: ["3.2", "6.7", "8.9", "9.12", "15.10", "16.9"]
TEST_SOLUTIONS: List[str] = ["3.2", "6.7", "8.9", "9.12", "15.10", "16.9"]
# output folder with no trailing slash
OUTPUT_FOLDER = "completions/all_finetuned"
DATA_JSON_FILE = 'All Data/data.json'

TEMPERATURE = 0.0 # between 0 and 1 (0 is the most predictable, 1 is the most surprising)
MAX_TOKENS = 1000 # the maximum amount of tokens the model can use for a completion
TOP_P = 1 # diversity via nucleus sampling (IDK what this means - just keep it at 1)
FREQUENCY_PENALTY = 0 # penalize new tokens based on their existing frequency in the text so far
PRESENCE_PENALTY = 0 # penalize new tokens based on whether (or not - it's binary) they appear in the text so far

# for grabbing the data out of the json file
USAGE_KEY = "usage"
TOTAL_TOKENS_KEY = "total_tokens"
CHOICES_KEY = "choices"
TEXT_KEY = "text"
FIRST_RESPONSE_KEY = 0

class Model(Enum):
    DAVINCI = "text-davinci-002"
    NEW_DAVINCI = "text-davinci-003"
    OLD_FINETUNED = OLD_FINETUNED_MODEL
    NEW_FINETUNED = NEW_FINETUNED_MODEL

class Prompt(Enum):
    DEFAULT_PROMPT = "Code the answer to the following question in Python.\n\nDataset: {dataset}\n\nQuestion: {question}\n\nCode:"
    FINETUNED_PROMPT = "{dataset}\n\nQuestion: {question}\n\nCode:"
    SHUAI = "Write python code for the following questions sentence by sentence.\n\nDataset: {dataset}\n\nQuestion: {question}\n\nCode:"
    HEMU = "Hi Sara, \n\n I was wondering if you could help me with this problem, as you are an expert in Machine Learning. I want to use this dataset: {dataset} to answer this question: {question}\n\n Can you give me the results in code?"
    STEP_BY_STEP = "Code the answer to the following question in Python.\n\nDataset: {dataset}\n\nQuestion: {question}\n\nCode: Let's think step by step."
    IMPROVED_STEP_BY_STEP = "Write python code for the following questions.\n\nDataset: {dataset}\n\nQuestion: {question}\n\nCode: Let's think step by step."

def num_to_sol(num):
    return f"sol{num}.json"

def sol_to_num(sol):
    return sol.replace("sol", "").replace(".json", "")

def make_path(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

prompt: Prompt = Prompt.FINETUNED_PROMPT
model: Model = Model.NEW_FINETUNED

if model == Model.NEW_FINETUNED or model == Model.OLD_FINETUNED:
    cost_per_token = 0.12 / 1000.0 # the price is 12 cents per thousand tokens for fine-tuned davinci (see https://openai.com/api/pricing/)
else:
    cost_per_token = 0.02 / 1000.0 # the price is 2 cents per thousand tokens for davinci (see https://openai.com/api/pricing/)


# load the json file into a list of dictionaries called examples
# make sure we use the right encoding
with open(DATA_JSON_FILE, 'r', encoding='utf-8') as f:
    examples = json.load(f)

# split into train and test using TEST_SOLUTIONS
test_solutions = [num_to_sol(num) for num in TEST_SOLUTIONS]
train_examples = []
test_examples = []
for example in examples:
    if example['code'] in test_solutions:
        test_examples.append(example)

test_examples_nums = [sol_to_num(example['code']) for example in test_examples]


assert isinstance(prompt, Prompt), f"prompt must be of type Prompt, not {type(prompt)}"
assert isinstance(model, Model), f"model must be of type Model, but is {type(model)}"
assert test_examples_nums.sort() == TEST_SOLUTIONS.sort(), f"test_examples does not match TEST_SOLUTIONS:\n{[sol_to_num(example['code']) for example in test_examples]}\n{TEST_SOLUTIONS}"

# confirm settings before actually running
print(f"Running with the following settings:\n\tmodel: {model}\n\tprompt: {prompt}\n\toutput_folder: {OUTPUT_FOLDER}\n\ttest_solutions: {test_solutions}\n\tdata_json_file: {DATA_JSON_FILE}\n\ttemperature: {TEMPERATURE}\n\tmax_tokens: {MAX_TOKENS}\n\ttop_p: {TOP_P}\n\tfrequency_penalty: {FREQUENCY_PENALTY}\n\tpresence_penalty: {PRESENCE_PENALTY}\n\tcost_per_token: {cost_per_token}")
input("Press enter to continue... (ctrl-c to quit)")

# for each of the test examples, get the responses from openai.
price = 0
print("Please note that outputs may not run in order, but all outputs will be generated eventually.")
for example in test_examples:
    sol_num = sol_to_num(example['code'])
    print(f"Getting completion for solution {sol_num}...")
    str_model = model.value
    str_prompt = prompt.value.format(dataset=example["dataset description"], question=example["question"])
    response = openai.Completion.create(
            model=str_model,
            prompt=str_prompt,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            frequency_penalty=FREQUENCY_PENALTY,
            presence_penalty=PRESENCE_PENALTY
        )
    text = response[CHOICES_KEY][FIRST_RESPONSE_KEY][TEXT_KEY]
    token_amount = response[USAGE_KEY][TOTAL_TOKENS_KEY]
    cost = token_amount * cost_per_token
    price += cost
    print(f"Current running price total is: {price}")

    # save the output to a file
    path = f"{OUTPUT_FOLDER}/completion_{sol_num}.txt"
    make_path(path)
    with open(path, 'w') as f:
        f.write(text)
    print(f"Output saved to {path}")