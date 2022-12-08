# 5525-final-project by Team 11
Shuai An, Hemu Kumar, London Lowmanstone

## How to Run
First install `requirements.txt`
To run predictions, create a folder called `secret` with a file called `secret.env` inside.
Inside of `secret.env`, set `OPENAI_API_KEY=` to be your OpenAI Key.
Additionally, set
`FINETUNED_MODEL=davinci:ft-university-of-minnesota-2022-10-20-09-05-28`
`NEW_FINETUNED_MODEL=davinci:ft-university-of-minnesota-2022-12-04-13-48-56`.

Finally, run `get_code_completion.py`, setting the variables in the code as needed.
The code will not run models or charge you unless you give it your consent by pressing enter to confirm your settings at the command prompt after starting the script.