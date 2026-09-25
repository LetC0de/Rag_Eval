import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.llms.openai_model import OpenAIModel
from deepeval.evaluate.configs import CacheConfig, ErrorConfig
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

# repo root on sys.path so `src` works whether run from root or from evals/
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.rag_pipeline import RagPipeline

load_dotenv()

GOLDEN_PATH = str(ROOT_DIR / "goldens" / "correctness_goldens.json")  # question + ideal_answer
JUDGE_MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b:free"    # nvidia/nemotron-3-super-120b-a12b:free
JUDGE_MODEL = OpenAIModel(
    model=JUDGE_MODEL_NAME,
    api_key=os.getenv("API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    generation_kwargs={
        "extra_body": {"reasoning": {"enabled": False}},
    },
)

JUDGE_MODEL.model_data.supports_json = True
JUDGE_MODEL.model_data.supports_structured_outputs = False

THRESHOLD = 0.7


# 1. LOAD queries + ideal answers (ideal_answer is the CORRECT answer, our reference)
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


# 2. RUN THE FULL PIPELINE per query, build a test case from LIVE output
rag = RagPipeline()
test_cases = []

for g in goldens[:1]:
    result = rag.invoke(g["question"])          # retrieve → rerank → generate

    test_cases.append(
        LLMTestCase(
            input=g["question"],
            actual_output=result["answer"],      # what the generator produced
            expected_output=g["ideal_answer"],   # the CORRECT reference answer
        )
    )


# 3c. STYLE – reference-free, judges TONE only (note: no EXPECTED_OUTPUT)
style = GEval(
    name="Style",
    evaluation_steps=[
        "Judge only the teaching style and tone of the actual output, not whether it is factually correct or complete.",
        "Reward an intuitive, explanatory tone: plain language, the idea explained before any formula or jargon, and technical terms briefly unpacked when used.",
        "Reward a direct, conversational register written in prose, as a CampusX lecture would explain it out loud, rather than a dry, formal, or bullet-list tone.",
        "An analogy or concrete example is a BONUS when the concept is abstract, but a clear, direct, well-explained answer is fully acceptable and must NOT be penalized for not having one.",
        "Penalize answers that are stiff, bureaucratic, structured as a bare list with no explanation, or that use unexplained jargon.",
        "Do NOT reward or penalize based on correctness, completeness, or length – only on style and tone.",
    ],
    rubric=[
        Rubric(
            score_range=(9, 10),
            expected_outcome="Clearly in a CampusX teaching voice: intuitive, conversational prose that explains before it formalizes.",
        ),
        Rubric(
            score_range=(7, 8),
            expected_outcome="Clear, conversational, and well-explained in prose. Fully acceptable even without an analogy or example.",
        ),
        Rubric(
            score_range=(4, 6),
            expected_outcome="Understandable but somewhat flat, formal, or list-heavy in places.",
        ),
        Rubric(
            score_range=(0, 3),
            expected_outcome="Dry, stiff, bare-list, jargon-heavy, or robotic; does not read like a teaching explanation.",
        ),
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,
)
# 4. EVALUATE
evaluate(
    test_cases=test_cases,
    metrics=[style],
    cache_config=CacheConfig(write_cache=False, use_cache=False),
    error_config=ErrorConfig(ignore_errors=True),
    hyperparameters={
        "retriever": "rerank_fetch10_k5",
        "embedding_model": "mistral-embed",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "judge_model": JUDGE_MODEL_NAME,
        "golden_set": GOLDEN_PATH,
    },
)