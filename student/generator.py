import os
from pathlib import Path
from typing import Dict, List

from llama_cpp import Llama

from .models import MinimalSource

MODEL_REPO = "Qwen/Qwen3-0.6B-GGUF"
MODEL_FILE = "*.gguf"
MAX_NEW_TOKENS = 200
N_CTX = 4096


class Generator:
	"""Generate answers using a local quantized LLM (GGUF via llama.cpp)."""

	def __init__(self) -> None:
		"""Load the quantized model with llama.cpp."""
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
		max_context_length: int = 1500,
	) -> str:
		"""Generate an answer grounded in the retrieved source chunks.

		Args:
			question: The user question.
			sources: Retrieved source chunks (in ranked order).
			max_context_length: Max total characters of context to include.

		Returns:
			The generated answer string.
		"""
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
				if total + len(content) > max_context_length:
					remaining = max_context_length - total
					if remaining > 200:
						content = content[:remaining]
					else:
						break
				context_parts.append(
					f"[Source: {src.file_path} chars {src.first_character_index}-"
					f"{src.last_character_index}]\n{content}"
				)
				total += len(content)
			except Exception:
				continue

		context = "\n\n---\n\n".join(context_parts)

		system_prompt = (
			"Answer the question concisely in 1-3 sentences using only the "
			"following context. Cite the specific name, command, value, or "
			"identifier from the context that answers the question; do not "
			"merely restate the question.\n\n"
			f"Context:\n{context}"
		)
		# Pre-fill an empty <think> block so Qwen3 skips its reasoning phase
		prompt = (
			f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
			f"<|im_start|>user\n{question}<|im_end|>\n"
			f"<|im_start|>assistant\n<think>\n\n</think>\n"
		)

		response = self._llm.create_completion(
			prompt=prompt,
			max_tokens=MAX_NEW_TOKENS,
			temperature=0.0,
			stop=["<|im_end|>"],
		)
		answer = response["choices"][0]["text"]
		return answer.strip()
