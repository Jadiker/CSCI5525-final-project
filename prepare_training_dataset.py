'''
Create a dataset for training OpenAI models.
The dataset needs to be in the form

```
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
...
```
'''

import json
import os
from tqdm import tqdm
from json_to_code import json_to_code

PROMPT = "{dataset}\n\nQuestion: {question}\n\nCode:"
CODE_DIRECTORY= "All Data"
DATA_JSON_FILE = "All Data/data.json"

TEST_SOLUTIONS = [3.2, 6.7, 8.9, 9.12, 15.10, 16.9]
EXPECTED_NUMBER_OF_TRAIN_SAMPLES = 27

def num_to_sol(num):
    return f"sol{num}.json"

def sol_to_num(sol):
    return sol.replace("sol", "").replace(".json", "")

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
    else:
        train_examples.append(example)

if len(train_examples) != EXPECTED_NUMBER_OF_TRAIN_SAMPLES:
    print(len(train_examples))
    print(len(test_examples))
    print(len(examples))
assert len(train_examples) == EXPECTED_NUMBER_OF_TRAIN_SAMPLES, f"Expected {EXPECTED_NUMBER_OF_TRAIN_SAMPLES} samples but found {len(train_examples)}: {[train_examples[i]['code'] for i in range(len(train_examples))]}"

# write the train_examples into the file
with open('train_dataset.jsonl', 'w', encoding='utf-8') as f:
    for example in tqdm(train_examples):
        answer = json_to_code(os.path.join(CODE_DIRECTORY, example['code']))
        prompt = PROMPT.format(dataset=example["dataset description"], question=example["question"])
        new_line = json.dumps({"prompt": prompt, "completion": " " + answer + " END"}) + "\n"
        f.write(new_line)

