import re
from typing import Dict, List, Set

SYNONYMS: Dict[str, List[str]] = {
    "k8s": ["Kubernetes"],
    "kubernetes": ["k8s"],
    "rocm": ["AMD"],
    "amd": ["ROCm"],
    "gaudi": ["Intel", "HPU"],
    "hpu": ["Gaudi"],
    "xpu": ["Intel"],
    "tpu": ["Google"],
    "neuron": ["AWS", "Trainium", "Inferentia"],
    "tp": ["tensor parallel", "tensor_parallel"],
    "pp": ["pipeline parallel", "pipeline_parallel"],
    "dp": ["data parallel", "data_parallel"],
    "ep": ["expert parallel", "expert_parallel"],
    "moe": ["mixture of experts", "expert parallel"],
    "openai-compatible": ["openai_compatible", "openai compatible"],
    "openai_compatible": ["openai-compatible"],
    "endpoint": ["url", "route"],
    "kv-cache": ["kvcache", "kv cache", "key-value cache"],
    "kvcache": ["kv-cache", "kv cache"],
    "pagedattention": ["paged attention", "paged-attention"],
    "prefix-caching": ["prefix_caching", "prefix caching"],
    "skypilot": ["sky pilot"],
    "tool-call": ["tool_call", "function calling", "function-calling"],
    "tool_call": ["tool-call", "function calling"],
    "yaml": ["yml"],
    "yml": ["yaml"],
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def expand_query(query: str) -> str:
    if not query.strip():
        return query

    tokens = _WORD_RE.findall(query)
    seen: Set[str] = {tok.lower() for tok in tokens}
    extras: List[str] = []

    for token in tokens:
        synonyms = SYNONYMS.get(token.lower(), [])
        for syn in synonyms:
            key = syn.lower()
            if key in seen:
                continue
            seen.add(key)
            extras.append(syn)

    if not extras:
        return query
    return f"{query} {' '.join(extras)}"
