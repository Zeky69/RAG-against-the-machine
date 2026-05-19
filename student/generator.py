import os
from pathlib import Path
from typing import Dict, List, cast

from llama_cpp import CreateCompletionResponse, Llama

from .models import MinimalSource

MODEL_REPO = "Qwen/Qwen3-0.6B-GGUF"
MODEL_FILE = "*.gguf"
MAX_NEW_TOKENS = 256
N_CTX = 4096
MAX_CONTEXT_CHARS = 3000
MAX_CHARS_PER_SOURCE = 900


class Generator:

    def __init__(self) -> None:
        self._llm = Llama.from_pretrained(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            n_ctx=N_CTX,
            n_threads=os.cpu_count() or 4,
            n_batch=512,
            verbose=False,
        )

    def generate(
            self,
            question: str,
            sources: List[MinimalSource],
            max_context_length: int = MAX_CONTEXT_CHARS,
    ) -> str:
        context_parts: List[str] = []
        total = 0
        file_cache: Dict[str, str] = {}
        for src in sources:
            try:
                if src.file_path not in file_cache:
                    file_cache[src.file_path] = Path(src.file_path).read_text(
                        encoding="utf-8", errors="ignore"
                    )
                content = file_cache[src.file_path][
                    src.first_character_index:src.last_character_index
                ]
                if len(content) > MAX_CHARS_PER_SOURCE:
                    content = content[:MAX_CHARS_PER_SOURCE].rstrip() + " …"
                if total + len(content) > max_context_length:
                    remaining = max_context_length - total
                    if remaining > 200:
                        content = content[:remaining]
                    else:
                        break
                context_parts.append(
                    f"[Source: {src.file_path} chars "
                    f"{src.first_character_index}-"
                    f"{src.last_character_index}]\n{content}"
                )
                total += len(content)
            except Exception:
                continue

        context = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are answering a technical question about the vLLM codebase. "
            "Use ONLY the context below to write a self-contained, thorough "
            "answer in 2-4 sentences, followed by a source citation. "
            "Requirements:\n"
            "  - State the direct answer first (name, command, value, or "
            "identifier), quoting the exact form from the context.\n"
            "  - Then add one sentence of relevant detail from the context "
            "(parameters, defaults, constraints, related options).\n"
            "  - If the context lists multiple items, name at least two.\n"
            "  - Do not restate the question; do not invent facts absent "
            "from the context.\n"
            "  - End your reply with a final line in the form: "
            "\"Source: <file_path>\" where <file_path> is the file the answer "
            "comes from.\n\n"
            f"Context:\n{context}"
        )
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n"
        )

        response = cast(
            CreateCompletionResponse,
            self._llm.create_completion(
                prompt=prompt,
                max_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                stop=["<|im_end|>"],
                stream=False,
            ),
        )
        answer = response["choices"][0]["text"]
        return answer.strip()
