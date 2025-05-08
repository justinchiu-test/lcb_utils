"""Utilities for computing similarity between prompts."""

import os
import datasets
from fuzzywuzzy import fuzz
import multiprocessing
import numpy as np
import pickle
from tqdm import tqdm
from typing import List, Tuple, Any, Dict, Optional
import argparse

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


def compute_similarity_matrix(
    prompts1: List[str], 
    prompts2: List[str], 
    processes: int = 128,
    output_file: Optional[str] = None
) -> np.ndarray:
    """Compute similarity matrix between two sets of prompts.
    
    Args:
        prompts1: First list of prompts
        prompts2: Second list of prompts
        processes: Number of processes to use for multiprocessing
        output_file: Optional path to save the similarity matrix pickle file
        
    Returns:
        numpy.ndarray: Similarity matrix with shape (len(prompts1), len(prompts2))
    """
    print(f"Computing similarity matrix between {len(prompts1)} and {len(prompts2)} prompts")
    
    # Create similarity matrix
    similarity_matrix = np.zeros((len(prompts1), len(prompts2)))

    # Prepare arguments for multiprocessing
    args_list = [(i, prompt, prompts2) for i, prompt in enumerate(prompts1)]

    # Use multiprocessing to compute similarities
    print("Computing similarities using multiprocessing...")
    with multiprocessing.Pool(processes=processes) as pool:
        results = list(
            tqdm(pool.imap(process_ocr_prompt, args_list), total=len(args_list))
        )

    # Fill the similarity matrix with results
    for idx1, similarities in results:
        similarity_matrix[idx1] = similarities

    # Save the similarity matrix if output file provided
    if output_file:
        with open(output_file, "wb") as f:
            pickle.dump(similarity_matrix, f)
        print(f"Saved similarity matrix to {output_file}")

    return similarity_matrix


def find_high_similarity_matches(
    prompts1: List[str],
    prompts2: List[str],
    similarity_matrix: np.ndarray,
    threshold: float = 90.0
) -> List[Tuple[int, int, float, str, str]]:
    """Find matches with high similarity scores.
    
    Args:
        prompts1: First list of prompts
        prompts2: Second list of prompts
        similarity_matrix: Similarity matrix with shape (len(prompts1), len(prompts2))
        threshold: Similarity threshold (default: 90.0)
        
    Returns:
        List of tuples (idx1, idx2, similarity, prompt1, prompt2) sorted by similarity
    """
    high_similarities = []

    for idx1, prompt1 in enumerate(prompts1):
        for idx2, prompt2 in enumerate(prompts2):
            similarity = similarity_matrix[idx1][idx2]
            if similarity >= threshold:
                high_similarities.append(
                    (idx1, idx2, similarity, prompt1, prompt2)
                )

    # Sort by similarity score in descending order
    high_similarities.sort(key=lambda x: x[2], reverse=True)
    
    return high_similarities


def main():
    """Command line interface for similarity computation."""
    parser = argparse.ArgumentParser(description="Compute similarities between two datasets")
    parser.add_argument("--lcb-dataset", default="livecodebench/code_generation_lite", 
                        help="LiveCodeBench dataset name")
    parser.add_argument("--lcb-version", default="v1_v3", help="LiveCodeBench dataset version")
    parser.add_argument("--ocr-dataset", default="nvidia/OpenCodeReasoning", 
                        help="OpenCodeReasoning dataset name")
    parser.add_argument("--ocr-split", default="split_0", help="OpenCodeReasoning split name")
    parser.add_argument("--processes", type=int, default=128, help="Number of processes to use")
    parser.add_argument("--threshold", type=float, default=90.0, 
                        help="Similarity threshold for high matches")
    parser.add_argument("--output", default="similarity_matrix.pkl", 
                        help="Output file for similarity matrix")
    
    args = parser.parse_args()
    
    # Load LCB datasets
    print("Loading LiveCodeBench dataset...")
    lcb_dataset = datasets.load_dataset(
        args.lcb_dataset,
        version_tag=args.lcb_version,
        trust_remote_code=True,
    )
    lcb_prompts = lcb_dataset["test"]["question_content"]

    # Load OCR dataset
    print("Loading OpenCodeReasoning dataset...")
    ocr_dataset = datasets.load_dataset(args.ocr_dataset, args.ocr_split)
    ocr_prompts = ocr_dataset[args.ocr_split]["input"]

    print(f"Number of LCB prompts: {len(lcb_prompts)}")
    print(f"Number of OCR prompts: {len(ocr_prompts)}")

    # Compute similarities
    similarity_matrix = compute_similarity_matrix(
        ocr_prompts, 
        lcb_prompts,
        processes=args.processes,
        output_file=args.output
    )
    
    # Find high similarity matches
    high_similarities = find_high_similarity_matches(
        ocr_prompts,
        lcb_prompts,
        similarity_matrix,
        threshold=args.threshold
    )

    # Print high similarity matches
    print(f"\nFound {len(high_similarities)} matches with similarity >= {args.threshold}%")
    for ocr_idx, lcb_idx, similarity, ocr_prompt, lcb_prompt in high_similarities[:20]:  # Show top 20
        print(f"Similarity: {similarity:.2f}%")
        print(f"OCR[{ocr_idx}]: {ocr_prompt[:100]}...")
        print(f"LCB[{lcb_idx}]: {lcb_prompt[:100]}...")
        print("-" * 80)


if __name__ == "__main__":
    main()
