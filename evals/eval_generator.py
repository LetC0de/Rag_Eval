"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault — the context was already correct.

    python -m evals.eval_generator
"""

import os
import json
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.models.llms.openai_model import OpenAIModel
from deepeval.evaluate.configs import CacheConfig, ErrorConfig
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from src.generator import generate   # your generator: generate(query, context) -> answer

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL_NAME = "nvidia/nemotron-3.5-lightning:free"
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


# 1. LOAD the faithfulness golden set (query + ideal_context)
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


# 2. RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
test_cases = []
for g in goldens[:10]:
    context = g["ideal_context"]              # known-good context (list of chunk strings)
    answer = generate(g["query"], context)    # RUN the generator -> actual_output

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=answer,             # the generated answer we're judging
            retrieval_context=context,        # faithfulness checks the answer against THIS
            # no expected_output — faithfulness never reads it
        )
    )


# 3. THE METRIC — decomposes actual_output into claims, attributes each to context
metrics = [FaithfulnessMetric(
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    include_reason=False,   # prints WHY each score — shows which claims were unsupported
),
AnswerRelevancyMetric(
    threshold=THRESHOLD, 
    model=JUDGE_MODEL, 
    include_reason=False)
]


# 4. EVALUATE — runs the metric on every case, prints a report
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    cache_config=CacheConfig(write_cache=False, use_cache=False),
    error_config=ErrorConfig(ignore_errors=True),
    hyperparameters={
        "retriever": "base_k5",
        "embedding_model": "text-embedding-3-small",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "judge_model": JUDGE_MODEL_NAME,
        "golden_set": GOLDEN_PATH,
    },
)