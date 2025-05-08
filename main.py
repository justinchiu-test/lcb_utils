import os
import datasets
from fuzzywuzzy import fuzz
import multiprocessing
import numpy as np
import pickle
from tqdm import tqdm

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def fuzzy_match(str1, str2):
    # Preprocess strings (optional)
    str1 = str1.strip().lower()
    str2 = str2.strip().lower()
    # Calculate similarity ratio (0-100)
    return fuzz.ratio(str1, str2)


def process_ocr_prompt(args):
    ocr_idx, ocr_prompt, lcb_prompts = args
    similarities = []
    for lcb_prompt in lcb_prompts:
        score = fuzzy_match(ocr_prompt, lcb_prompt)
        similarities.append(score)
    return ocr_idx, similarities


def main():
    # Load LCB datasets
    print("Loading LiveCodeBench dataset...")
    lcb13 = datasets.load_dataset(
        "livecodebench/code_generation_lite",
        version_tag="v1_v3",
        trust_remote_code=True,
    )
    lcb_prompts = lcb13["test"]["question_content"]

    # Load OCR dataset
    print("Loading OpenCodeReasoning dataset...")
    ocr0 = datasets.load_dataset("nvidia/OpenCodeReasoning", "split_0")
    ocr_prompts = ocr0["split_0"]["input"]

    print(f"Number of LCB prompts: {len(lcb_prompts)}")
    print(f"Number of OCR prompts: {len(ocr_prompts)}")

    # Create similarity matrix
    similarity_matrix = np.zeros((len(ocr_prompts), len(lcb_prompts)))

    # Prepare arguments for multiprocessing
    args_list = [(i, prompt, lcb_prompts) for i, prompt in enumerate(ocr_prompts)]

    # Use multiprocessing to compute similarities
    print("Computing similarities using multiprocessing...")
    with multiprocessing.Pool(processes=128) as pool:
        results = list(
            tqdm(pool.imap(process_ocr_prompt, args_list), total=len(args_list))
        )

    # Fill the similarity matrix with results
    for ocr_idx, similarities in results:
        similarity_matrix[ocr_idx] = similarities

    # Save the similarity matrix
    with open("similarity_matrix.pkl", "wb") as f:
        pickle.dump(similarity_matrix, f)

    # Find high similarity matches
    threshold = 90  # Similarity threshold
    high_similarities = []

    for ocr_idx, ocr_prompt in enumerate(ocr_prompts):
        for lcb_idx, lcb_prompt in enumerate(lcb_prompts):
            similarity = similarity_matrix[ocr_idx][lcb_idx]
            if similarity >= threshold:
                high_similarities.append(
                    (ocr_idx, lcb_idx, similarity, ocr_prompt, lcb_prompt)
                )

    # Sort by similarity score in descending order
    high_similarities.sort(key=lambda x: x[2], reverse=True)

    # Print high similarity matches
    print(f"\nFound {len(high_similarities)} matches with similarity >= {threshold}%")
    for ocr_idx, lcb_idx, similarity, ocr_prompt, lcb_prompt in high_similarities[
        :20
    ]:  # Show top 20
        print(f"Similarity: {similarity:.2f}%")
        print(f"OCR[{ocr_idx}]: {ocr_prompt[:100]}...")
        print(f"LCB[{lcb_idx}]: {lcb_prompt[:100]}...")
        print("-" * 80)

    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
