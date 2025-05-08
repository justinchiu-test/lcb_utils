"""Utilities for working with LiveCodeBench datasets."""

from lcb_utils.similarity import fuzzy_match, process_ocr_prompt
from lcb_utils.completions import get_completions_batch, get_completions_with_retry, RawCompletionResult
from lcb_utils.dataset import (
    Platform, 
    Difficulty, 
    TestType, 
    Test, 
    CodeGenerationProblem,
    Problem,
    load_code_generation_dataset,
    load_tests
)

__version__ = "0.1.0"