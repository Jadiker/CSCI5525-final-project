# currently, every example is a dictionary wrapped in a list.
# just remove that outer list

import json

with open('Even Chapters/data description.txt', 'r', encoding='utf-8') as f:
    examples = json.load(f)

print(examples[0])
new_examples = []
for example in examples:
    new_examples.append(example[0])
print(new_examples[0])
# save it
with open('Even Chapters/new data description.txt', 'w', encoding='utf-8') as f:
    json.dump(new_examples, f)