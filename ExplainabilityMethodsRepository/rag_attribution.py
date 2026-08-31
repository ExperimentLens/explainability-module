import re
from typing import Any, Dict, List, Optional, Tuple

from llmSHAP.data_handler import DataHandler
from llmSHAP.generation import Generation
from llmSHAP.prompt_codec import BasicPromptCodec, PromptCodec
from llmSHAP.attribution_methods.shapley_attribution import ShapleyAttribution

from ExplainabilityMethodsRepository.ollama_interface import OllamaInterface

SYSTEM_MESSAGE = "You are a grounded QA assistant."
QUESTION_KEY = "question"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def _token_similarity(a: str, b: str) -> float:
    a_tokens = set(_normalize(a).split())
    b_tokens = set(_normalize(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def chunk_label(index: int, source: str) -> str:
    return f"chunk_{index} ({source})"


def _format_context(chunks: List[str], sources: List[str]) -> str:
    return "\n\n".join(
        f"[{i}] source={source}\n{chunk}"
        for i, (chunk, source) in enumerate(zip(chunks, sources), start=1)
    )


def _instruction_from_template(prompt_template: str) -> str:
    return prompt_template.split("\n\nContext:")[0]


def _render_prompt(prompt_template: str, chunks: List[str], sources: List[str], question: str) -> str:
    context = _format_context(chunks, sources)
    return prompt_template.format(context=context, question=question)


def explain_chunks(
    question: str,
    retrieved_chunks: List[str],
    retrieved_sources: List[str],
    prompt_template: str,
    model_name: str,
    temperature: float,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    data = {
        chunk_label(i, source): chunk
        for i, (chunk, source) in enumerate(zip(retrieved_chunks, retrieved_sources))
    }
    instruction = _instruction_from_template(prompt_template)
    codec = BasicPromptCodec(system=instruction + "\n\nQuestion: " + question)
    handler = DataHandler(data)
    model = OllamaInterface(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

    result = ShapleyAttribution(
        model=model,
        data_handler=handler,
        prompt_codec=codec,
        use_cache=True,
        num_threads=8,
    ).attribution()

    return {
        "output": result.output,
        "attribution": result.attribution,
        "heatmap": result.render(),
    }


def counterfactual_check(
    question: str,
    retrieved_chunks: List[str],
    retrieved_sources: List[str],
    prompt_template: str,
    model_name: str,
    temperature: float,
    top_chunk_index: int,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    model = OllamaInterface(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

    original_prompt = _render_prompt(prompt_template, retrieved_chunks, retrieved_sources, question)

    cf_chunks = [c for i, c in enumerate(retrieved_chunks) if i != top_chunk_index]
    cf_sources = [s for i, s in enumerate(retrieved_sources) if i != top_chunk_index]
    counterfactual_prompt = _render_prompt(prompt_template, cf_chunks, cf_sources, question)

    def _ask(user_content: str) -> str:
        return model.generate(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_content},
            ]
        )

    original_answer = _ask(original_prompt)
    counterfactual_answer = _ask(counterfactual_prompt)

    changed = _normalize(original_answer) != _normalize(counterfactual_answer)
    similarity_score = _token_similarity(original_answer, counterfactual_answer)

    return {
        "original_answer": original_answer,
        "counterfactual_answer": counterfactual_answer,
        "changed": changed,
        "similarity_score": similarity_score,
    }


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


class SentenceCodec(PromptCodec):
    """Feeds the reconstructed sentence selection as a single user message (no
    separate system prompt) - the instruction itself is one of the attributable
    sentences here, unlike the chunk-level BasicPromptCodec where it's fixed."""

    def build_prompt(self, data_handler: DataHandler, indexes) -> Any:
        text = data_handler.to_string(indexes, mask=False, exclude_permanent_keys=False)
        return [{"role": "user", "content": text}]

    def parse_generation(self, model_output: str) -> Generation:
        return Generation(output=model_output)


def explain_sentences(
    question: str,
    retrieved_chunks: List[str],
    retrieved_sources: List[str],
    prompt_template: str,
    model_name: str,
    temperature: float,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Sentence-level Shapley attribution: every sentence in the instruction and in each
    retrieved chunk is its own feature; the question is pinned via permanent_keys (always
    present, never masked/scored) since its necessity isn't in question."""
    instruction = _instruction_from_template(prompt_template)

    data: Dict[str, str] = {}
    # chunk_index is None for instruction/question sentences (no originating chunk to edit).
    origin: Dict[str, Tuple[Optional[int], str]] = {}

    for j, sentence in enumerate(_split_sentences(instruction)):
        key = f"instruction s{j}"
        data[key] = sentence
        origin[key] = (None, "instruction")

    for i, (chunk, source) in enumerate(zip(retrieved_chunks, retrieved_sources)):
        for j, sentence in enumerate(_split_sentences(chunk)):
            key = f"{chunk_label(i, source)} s{j}"
            data[key] = sentence
            origin[key] = (i, source)

    data[QUESTION_KEY] = question
    origin[QUESTION_KEY] = (None, "question")

    handler = DataHandler(data, permanent_keys={QUESTION_KEY})
    model = OllamaInterface(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

    result = ShapleyAttribution(
        model=model,
        data_handler=handler,
        prompt_codec=SentenceCodec(),
        use_cache=True,
        num_threads=8,
    ).attribution()

    return {
        "output": result.output,
        "attribution": result.attribution,
        "heatmap": result.render(),
        "origin": origin,
    }


def counterfactual_check_sentence(
    question: str,
    retrieved_chunks: List[str],
    retrieved_sources: List[str],
    prompt_template: str,
    model_name: str,
    temperature: float,
    top_key: str,
    origin: Dict[str, Tuple[Optional[int], str]],
    max_tokens: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Counterfactual for the top-attributed sentence: strip just that sentence out of its
    originating chunk (or the instruction) and re-render. Returns None for the pinned
    question key - it can't be meaningfully removed and still ask something coherent."""
    chunk_index, place = origin.get(top_key, (None, None))
    if place == "question":
        return None

    model = OllamaInterface(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    original_prompt = _render_prompt(prompt_template, retrieved_chunks, retrieved_sources, question)

    if place == "instruction":
        instruction = _instruction_from_template(prompt_template)
        sentence_index = int(top_key.rsplit("s", 1)[1])
        remaining = [
            s for j, s in enumerate(_split_sentences(instruction)) if j != sentence_index
        ]
        modified_template = " ".join(remaining) + "\n\nContext:\n{context}\n\nQuestion: {question}"
        counterfactual_prompt = _render_prompt(
            modified_template, retrieved_chunks, retrieved_sources, question
        )
    else:
        sentence_index = int(top_key.rsplit("s", 1)[1])
        modified_chunks = list(retrieved_chunks)
        remaining = [
            s
            for j, s in enumerate(_split_sentences(modified_chunks[chunk_index]))
            if j != sentence_index
        ]
        modified_chunks[chunk_index] = " ".join(remaining)
        counterfactual_prompt = _render_prompt(
            prompt_template, modified_chunks, retrieved_sources, question
        )

    def _ask(user_content: str) -> str:
        return model.generate(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_content},
            ]
        )

    original_answer = _ask(original_prompt)
    counterfactual_answer = _ask(counterfactual_prompt)

    changed = _normalize(original_answer) != _normalize(counterfactual_answer)
    similarity_score = _token_similarity(original_answer, counterfactual_answer)

    return {
        "original_answer": original_answer,
        "counterfactual_answer": counterfactual_answer,
        "changed": changed,
        "similarity_score": similarity_score,
    }
