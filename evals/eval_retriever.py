import json
import os

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models.llms.openai_model import OpenAIModel
from deepeval.evaluate.configs import CacheConfig, ErrorConfig

from src.retriever import build_retriever

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"
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
# OpenRouter custom model not in deepeval's OPENAI_MODELS_DATA -> force JSON mode
# otherwise deepeval does plain generation and brittle trimAndLoadJson fails on reasoning traces
JUDGE_MODEL.model_data.supports_json = True
JUDGE_MODEL.model_data.supports_structured_outputs = False

THRESHOLD = 0.7


# 1. LOAD the golden set --- the fixed, human-authored truth
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


# 2. RUN THE RETRIEVER on each question to fill retrieval_context,
#    then build one test case per golden.
retriever = build_retriever()

test_cases = []

for g in goldens:
    retrieved = retriever.invoke(g["query"])
    retrieval_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)",
        )
    )


# 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
metrics = [
    ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]


# 4. EVALUATE
# cache disabled = fix for Windows portalocker bug (AttributeError: test_cases_lookup_map)
# error_config = one bad JSON verdict doesn't crash entire 15-case run
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
