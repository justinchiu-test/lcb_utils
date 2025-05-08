import datasets
import json
import os


# Set environment variables
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

dataset = datasets.load_dataset(
    "livecodebench/code_generation_lite",
    version_tag="v1_v3",
    trust_remote_code=True,
    split="test",
)
question2id = {x["question_content"]: x["question_id"] for x in dataset}

with open("labelled_data/samples_with_correct_r1_solutions.jsonl", "r") as f:
    prompts = [json.loads(line)["question"] for line in f]

successful_ids = [question2id[prompt] for prompt in prompts]

with open("labelled_data/successful_ids.txt", "w") as f:
    for id in successful_ids:
        f.write(id)
        f.write("\n")
