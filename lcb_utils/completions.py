"""Utilities for generating LLM completions using the Together API."""

import os
import json
import asyncio
import datasets
from tqdm.asyncio import tqdm_asyncio
import together
from together import AsyncTogether
from together.types.common import UsageData
from typing import List, Dict, Any, Optional, Set
import argparse
import time
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)

# Set environment variables
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Default configuration
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1"
DEFAULT_MAX_TOKENS = 18000
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_CONCURRENCY = 10  # Maximum number of concurrent requests
DEFAULT_RATE_LIMIT_DELAY = 0.1  # Delay between API calls to avoid rate limiting
DEFAULT_NUM_COMPLETIONS_PER_PROMPT = 4  # Number of completions per prompt
DEFAULT_MAX_RETRIES = 3  # Maximum number of retry attempts
DEFAULT_MIN_RETRY_WAIT = 1  # Minimum wait time between retries in seconds
DEFAULT_MAX_RETRY_WAIT = 10  # Maximum wait time between retries in seconds

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("lcb_utils.completions")


# Define pydantic models for validation
class RawCompletionResult(BaseModel):
    """Raw completion result with all data needed for later processing"""

    prompt_id: str
    prompt: str
    completions: List[str] = Field(default_factory=list)
    usage: Optional[UsageData] = None
    model: str
    success: bool
    error: Optional[str] = None
    timestamp: str


async def _api_call(
    client: AsyncTogether,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    num_completions: int,
):
    """Make the actual API call to Together with retries"""
    return await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=num_completions,  # Request multiple completions
    )


# Define the retry decorator for the API call
@retry(
    stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=DEFAULT_MIN_RETRY_WAIT, max=DEFAULT_MAX_RETRY_WAIT),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def get_completions_with_retry(
    client: AsyncTogether,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    num_completions: int,
):
    """Get completions with retry logic"""
    return await _api_call(
        client=client,
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        num_completions=num_completions,
    )


async def get_completions_batch(
    client: AsyncTogether,
    prompt: str,
    prompt_id: str,
    model: str,
    jsonl_file: str,
    jsonl_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    num_completions: int = DEFAULT_NUM_COMPLETIONS_PER_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> RawCompletionResult:
    """Get multiple completions from the Together API in a single request and checkpoint to JSONL"""
    # Initialize result
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    result = RawCompletionResult(
        prompt_id=prompt_id,
        prompt=prompt,
        model=model,
        success=False,
        timestamp=timestamp,
    )

    async with semaphore:
        # Add small delay to avoid rate limiting
        await asyncio.sleep(DEFAULT_RATE_LIMIT_DELAY)

        try:
            # Get multiple completions with retry logic
            logger.info(f"Getting completions for prompt ID: {prompt_id}")
            response = await get_completions_with_retry(
                client=client,
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                num_completions=num_completions,
            )

            # Process successful response
            completions = [choice.message.content for choice in response.choices]

            # Update result
            result.completions = completions
            result.usage = response.usage
            result.success = True
            logger.info(
                f"Successfully got {len(completions)} completions for prompt ID: {prompt_id}"
            )

        except Exception as e:
            # Handle other API errors
            error_msg = str(e)
            logger.error(
                f"Error getting completions for prompt {prompt_id}: {error_msg}"
            )
            result.error = error_msg
            result.success = False

    # Checkpoint to JSONL file immediately
    async with jsonl_lock:
        with open(jsonl_file, "a") as f:
            f.write(json.dumps(result.model_dump()) + "\n")

    return result


async def generate_completions(
    api_key: str = None,
    output_dir: str = "completions",
    raw_jsonl: str = None,
    grouped_jsonl: str = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    model: str = DEFAULT_MODEL,
    num_completions: int = DEFAULT_NUM_COMPLETIONS_PER_PROMPT,
    start_idx: int = 0,
    end_idx: int = None,
    overwrite: bool = False,
    lcb_dataset: str = "livecodebench/code_generation_lite",
    lcb_version: str = "v1_v3"
) -> Dict[str, Any]:
    """Generate completions for the LiveCodeBench dataset using the Together API.
    
    Args:
        api_key: Together API key (defaults to TOGETHER_API_KEY environment variable)
        output_dir: Directory to save completions
        raw_jsonl: Path to raw JSONL output file (default: output_dir/raw_completions.jsonl)
        grouped_jsonl: Path to grouped JSONL output file (default: output_dir/completions.jsonl)
        max_concurrency: Maximum number of concurrent requests
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        model: Model to use for completions
        num_completions: Number of completions per prompt
        start_idx: Index to start processing from
        end_idx: Index to end processing at (exclusive)
        overwrite: Overwrite existing results for prompts that have already been processed
        lcb_dataset: LiveCodeBench dataset to use
        lcb_version: LiveCodeBench dataset version
    
    Returns:
        Dict containing summary statistics
    """
    # Ensure API key is set
    api_key = api_key or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not provided and TOGETHER_API_KEY environment variable not set. "
            "Please provide an API key or export TOGETHER_API_KEY=your_api_key"
        )
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set raw JSONL file path if not specified
    if raw_jsonl is None:
        raw_jsonl = os.path.join(output_dir, "raw_completions.jsonl")

    # Set grouped JSONL file path if not specified
    if grouped_jsonl is None:
        grouped_jsonl = os.path.join(output_dir, "completions.jsonl")
        
    return await _generate_completions_impl(
        api_key=api_key,
        output_dir=output_dir,
        raw_jsonl=raw_jsonl,
        grouped_jsonl=grouped_jsonl,
        max_concurrency=max_concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        model=model,
        num_completions=num_completions,
        start_idx=start_idx,
        end_idx=end_idx,
        overwrite=overwrite,
        lcb_dataset=lcb_dataset,
        lcb_version=lcb_version
    )


async def _generate_completions_impl(
    api_key: str,
    output_dir: str,
    raw_jsonl: str,
    grouped_jsonl: str,
    max_concurrency: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    model: str,
    num_completions: int,
    start_idx: int,
    end_idx: int,
    overwrite: bool,
    lcb_dataset: str,
    lcb_version: str
) -> Dict[str, Any]:
    """Implementation of generate_completions."""
    # Check if output files exist and load completed prompts
    completed_prompt_ids = set()
    if os.path.exists(raw_jsonl) and not overwrite:
        # Load previously completed prompt IDs
        try:
            with open(raw_jsonl, "r") as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("success", False):
                        completed_prompt_ids.add(data["prompt_id"])
            logger.info(f"Found {len(completed_prompt_ids)} already completed prompts")
            # Append mode for existing file
            raw_jsonl_mode = "a"
        except Exception as e:
            logger.warning(f"Error reading existing raw JSONL file: {e}")
            completed_prompt_ids = set()
            raw_jsonl_mode = "w"  # Write mode (overwrite) if error
    else:
        # Create/clear the JSONL files if overwrite is set or files don't exist
        raw_jsonl_mode = "w"

    # Create or truncate files as needed
    if raw_jsonl_mode == "w":
        # Create/clear the output files
        with open(raw_jsonl, "w") as f:
            pass  # Create empty file
        logger.info(f"Created or reset raw JSONL file: {raw_jsonl}")

    print(f"Raw completion results will be saved to: {raw_jsonl}")
    print(f"Grouped completions will be saved to: {grouped_jsonl}")
    if not overwrite and completed_prompt_ids:
        print(f"Skipping {len(completed_prompt_ids)} already completed prompts")

    # Load LCB dataset
    print(f"Loading LiveCodeBench dataset {lcb_dataset} with version {lcb_version}...")
    lcb_dataset_obj = datasets.load_dataset(
        lcb_dataset,
        version_tag=lcb_version,
        trust_remote_code=True,
    )

    # Get test prompts
    test_data = lcb_dataset_obj["test"]
    prompts = test_data["question_content"]
    prompt_ids = test_data["question_id"]

    # Set end index if not specified
    if end_idx is None:
        end_idx = len(prompts)

    # Select prompts to process
    selected_prompts = prompts[start_idx : end_idx]
    selected_ids = prompt_ids[start_idx : end_idx]
    num_prompts = len(selected_prompts)

    # Initialize shared resources
    semaphore = asyncio.Semaphore(max_concurrency)
    jsonl_lock = asyncio.Lock()
    client = AsyncTogether(api_key=api_key)

    # Update global constants if they were overridden by arguments
    logger.info(f"Using model: {model}")
    logger.info(f"Generating {num_completions} completions per prompt")
    logger.info(f"Max concurrency: {max_concurrency}")
    logger.info(f"Max retries per request: {DEFAULT_MAX_RETRIES}")

    # Create tasks for prompts that need processing
    tasks = []
    prompts_to_process = 0
    prompts_skipped = 0

    for i, (prompt, prompt_id) in enumerate(zip(selected_prompts, selected_ids)):
        # Skip prompts that have already been successfully processed
        if not overwrite and prompt_id in completed_prompt_ids:
            prompts_skipped += 1
            continue

        prompts_to_process += 1
        task = get_completions_batch(
            client=client,
            prompt=prompt,
            prompt_id=prompt_id,
            model=model,
            jsonl_file=raw_jsonl,
            jsonl_lock=jsonl_lock,
            semaphore=semaphore,
            num_completions=num_completions,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        tasks.append(task)

    print(
        f"Creating tasks for {prompts_to_process} prompts with {num_completions} completions each..."
    )
    if prompts_skipped > 0:
        print(f"Skipping {prompts_skipped} already completed prompts")

    # Process all tasks with progress tracking
    if tasks:
        start_time = time.time()
        print(f"Starting to process {prompts_to_process} prompts...")
        results = await tqdm_asyncio.gather(*tasks, desc="Getting completions")
    else:
        print("No prompts to process - all have been completed already")
        results = []
        start_time = (
            time.time()
        )  # Set start time to now for consistent summary generation

    # Calculate statistics for this run
    elapsed_time = time.time() - start_time
    new_successful = sum(1 for r in results if r.success)
    new_failed = len(results) - new_successful
    new_completions = sum(len(r.completions) for r in results if r.success)

    # Get total statistics including already completed prompts
    total_successful = len(completed_prompt_ids) + new_successful
    total_failed = num_prompts - total_successful

    # Calculate total completions by reading the raw file
    total_completions = 0
    try:
        with open(raw_jsonl, "r") as f:
            for line in f:
                data = json.loads(line)
                if data.get("success", False):
                    total_completions += len(data.get("completions", []))
    except Exception as e:
        logger.warning(f"Error counting total completions: {e}")
        total_completions = new_completions  # Fallback to just new completions

    # Create a summary file
    summary = {
        "total_prompts": num_prompts,
        "prompts_processed_this_run": len(results),
        "prompts_skipped_this_run": prompts_skipped,
        "total_completions_requested": num_prompts * num_completions,
        "total_completions_received": total_completions,
        "completions_per_prompt": num_completions,
        "successful_prompts": total_successful,
        "failed_prompts": total_failed,
        "model": model,
        "elapsed_time_this_run": elapsed_time,
        "average_time_per_prompt": elapsed_time / len(results)
        if len(results) > 0
        else 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_jsonl_file": raw_jsonl,
        "grouped_jsonl_file": grouped_jsonl,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print statistics for this run
    if prompts_to_process > 0:
        print(f"Completed processing {len(results)} prompts in {elapsed_time:.2f}s")
        print(
            f"Success rate this run: {new_successful}/{len(results)} prompts "
            f"({new_successful / len(results) * 100 if len(results) > 0 else 0:.2f}%)"
        )
        print(f"Generated {new_completions} new completions in this run")
        if len(results) > 0:
            print(
                f"Average time per prompt this run: {elapsed_time / len(results):.2f}s"
            )

    # Print overall statistics
    print(
        f"\nOverall progress: {total_successful}/{num_prompts} prompts completed "
        f"({total_successful / num_prompts * 100:.2f}%)"
    )
    print(f"Total completions across all runs: {total_completions}")
    print(f"Raw results saved to: {raw_jsonl}")

    # Create final grouped output in the desired format
    print("Creating grouped output file...")

    # Always reprocess the entire raw file to ensure the grouped output is complete
    grouped_data = {}
    try:
        # Read all successfully completed prompts from the raw file
        with open(raw_jsonl, "r") as f:
            for line in f:
                data = json.loads(line)
                if data.get("success", False):
                    grouped_data[data["prompt_id"]] = {
                        "id": data["prompt_id"],
                        "prompt": data["prompt"],
                        "completions": data["completions"],
                    }
    except Exception as e:
        logger.error(f"Error reading raw JSONL file for grouped output: {e}")

    # Also add any newly processed results
    for result in results:
        if result.success:
            grouped_data[result.prompt_id] = {
                "id": result.prompt_id,
                "prompt": result.prompt,
                "completions": result.completions,
            }

    # Sort results by prompt ID for consistency
    sorted_records = sorted(grouped_data.values(), key=lambda x: x["id"])

    # Write the grouped output
    with open(grouped_jsonl, "w") as f:
        for record in sorted_records:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(sorted_records)} records to grouped output file")
    print(f"Grouped results saved to: {grouped_jsonl}")
    
    return summary


async def main():
    """Command line interface for generating completions."""
    parser = argparse.ArgumentParser(
        description="Get completions from Together API for LCB dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="completions",
        help="Directory to save completions",
    )
    parser.add_argument(
        "--raw_jsonl",
        type=str,
        default=None,
        help="Path to raw JSONL output file (default: output_dir/raw_completions.jsonl)",
    )
    parser.add_argument(
        "--grouped_jsonl",
        type=str,
        default=None,
        help="Path to grouped JSONL output file (default: output_dir/completions.jsonl)",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=DEFAULT_TOP_P, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="Model to use for completions"
    )
    parser.add_argument(
        "--num_completions",
        type=int,
        default=DEFAULT_NUM_COMPLETIONS_PER_PROMPT,
        help="Number of completions per prompt",
    )
    parser.add_argument(
        "--start_idx", type=int, default=0, help="Index to start processing from"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="Index to end processing at (exclusive)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results for prompts that have already been processed",
    )
    parser.add_argument(
        "--lcb-dataset", 
        default="livecodebench/code_generation_lite",
        help="LiveCodeBench dataset to use"
    )
    parser.add_argument(
        "--lcb-version", 
        default="v1_v3",
        help="LiveCodeBench dataset version"
    )

    args = parser.parse_args()
    
    # Call the main implementation function
    await generate_completions(
        output_dir=args.output_dir,
        raw_jsonl=args.raw_jsonl,
        grouped_jsonl=args.grouped_jsonl,
        max_concurrency=args.max_concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        model=args.model,
        num_completions=args.num_completions,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        overwrite=args.overwrite,
        lcb_dataset=args.lcb_dataset,
        lcb_version=args.lcb_version
    )


if __name__ == "__main__":
    asyncio.run(main())