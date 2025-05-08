import os
import json
import asyncio
import datasets
from tqdm.asyncio import tqdm_asyncio
import together
from together import AsyncTogether
from together.types.common import UsageData
from typing import List, Dict, Any, Optional
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

# Configure Together API - make sure to set your API key as an environment variable
# export TOGETHER_API_KEY=your_api_key_here
API_KEY = os.environ.get("TOGETHER_API_KEY")
MODEL = "deepseek-ai/DeepSeek-R1"
MAX_TOKENS = 18000
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_CONCURRENCY = 10  # Maximum number of concurrent requests
RATE_LIMIT_DELAY = 0.1  # Delay between API calls to avoid rate limiting
NUM_COMPLETIONS_PER_PROMPT = 4  # Number of completions per prompt
MAX_RETRIES = 3  # Maximum number of retry attempts
MIN_RETRY_WAIT = 1  # Minimum wait time between retries in seconds
MAX_RETRY_WAIT = 10  # Maximum wait time between retries in seconds

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("get_completions")


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
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
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
    num_completions: int = NUM_COMPLETIONS_PER_PROMPT,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
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
        await asyncio.sleep(RATE_LIMIT_DELAY)

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


async def main():
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
        default=MAX_CONCURRENCY,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=MAX_TOKENS, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=TOP_P, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL, help="Model to use for completions"
    )
    parser.add_argument(
        "--num_completions",
        type=int,
        default=NUM_COMPLETIONS_PER_PROMPT,
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

    args = parser.parse_args()

    # Ensure API key is set
    if not API_KEY:
        raise ValueError(
            "TOGETHER_API_KEY environment variable not set. Please export TOGETHER_API_KEY=your_api_key"
        )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set raw JSONL file path if not specified
    if args.raw_jsonl is None:
        args.raw_jsonl = os.path.join(args.output_dir, "raw_completions.jsonl")

    # Set grouped JSONL file path if not specified
    if args.grouped_jsonl is None:
        args.grouped_jsonl = os.path.join(args.output_dir, "completions.jsonl")

    # Create/clear the JSONL file
    with open(args.raw_jsonl, "w") as f:
        pass  # Create empty file

    print(f"Raw completion results will be saved to: {args.raw_jsonl}")
    print(f"Grouped completions will be saved to: {args.grouped_jsonl}")

    # Load LCB dataset
    print("Loading LiveCodeBench dataset...")
    lcb_dataset = datasets.load_dataset(
        "livecodebench/code_generation_lite",
        version_tag="v1_v3",
        trust_remote_code=True,
    )

    # Get test prompts
    test_data = lcb_dataset["test"]
    prompts = test_data["question_content"]
    prompt_ids = test_data["question_id"]

    # Set end index if not specified
    if args.end_idx is None:
        args.end_idx = len(prompts)

    # Select prompts to process
    selected_prompts = prompts[args.start_idx : args.end_idx]
    selected_ids = prompt_ids[args.start_idx : args.end_idx]
    num_prompts = len(selected_prompts)

    # Initialize shared resources
    semaphore = asyncio.Semaphore(args.max_concurrency)
    jsonl_lock = asyncio.Lock()
    client = AsyncTogether(api_key=API_KEY)

    # Update global constants if they were overridden by arguments
    logger.info(f"Using model: {args.model}")
    logger.info(f"Generating {args.num_completions} completions per prompt")
    logger.info(f"Max concurrency: {args.max_concurrency}")
    logger.info(f"Max retries per request: {MAX_RETRIES}")

    # Create tasks for all prompts (one task per prompt)
    print(
        f"Creating tasks for {num_prompts} prompts with {args.num_completions} completions each..."
    )
    tasks = []
    for i, (prompt, prompt_id) in enumerate(zip(selected_prompts, selected_ids)):
        task = get_completions_batch(
            client=client,
            prompt=prompt,
            prompt_id=prompt_id,
            model=args.model,
            jsonl_file=args.raw_jsonl,
            jsonl_lock=jsonl_lock,
            semaphore=semaphore,
            num_completions=args.num_completions,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        tasks.append(task)

    # Process all tasks with progress tracking
    start_time = time.time()
    print(f"Starting to process {num_prompts} prompts...")
    results = await tqdm_asyncio.gather(*tasks, desc="Getting completions")

    # Calculate statistics
    elapsed_time = time.time() - start_time
    successful = sum(1 for r in results if r.success)
    failed = num_prompts - successful
    total_completions = sum(len(r.completions) for r in results if r.success)

    # Create a summary file
    summary = {
        "total_prompts": num_prompts,
        "total_completions_requested": num_prompts * args.num_completions,
        "total_completions_received": total_completions,
        "completions_per_prompt": args.num_completions,
        "successful_prompts": successful,
        "failed_prompts": failed,
        "model": args.model,
        "elapsed_time": elapsed_time,
        "average_time_per_prompt": elapsed_time / num_prompts if num_prompts > 0 else 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_jsonl_file": args.raw_jsonl,
        "grouped_jsonl_file": args.grouped_jsonl,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Completed processing {num_prompts} prompts in {elapsed_time:.2f}s")
    print(
        f"Success rate: {successful}/{num_prompts} prompts ({successful / num_prompts * 100:.2f}%)"
    )
    print(
        f"Generated {total_completions} completions out of {num_prompts * args.num_completions} requested"
    )
    print(f"Average time per prompt: {elapsed_time / num_prompts:.2f}s")
    print(f"Raw results saved to: {args.raw_jsonl}")

    # Create final grouped output in the desired format
    print("Creating grouped output file...")
    grouped_data = []
    for result in results:
        if not result.success:
            continue

        # Create the grouped record
        record = {
            "id": result.prompt_id,
            "prompt": result.prompt,
            "completions": result.completions,
        }
        grouped_data.append(record)

    # Write the grouped output
    with open(args.grouped_jsonl, "w") as f:
        for record in grouped_data:
            f.write(json.dumps(record) + "\n")

    print(f"Grouped results saved to: {args.grouped_jsonl}")


if __name__ == "__main__":
    asyncio.run(main())
