import os
from pathlib import Path
from typing import Dict, List

from llama_cpp import Llama

from .models import MinimalSource

MODEL_REPO = "Qwen/Qwen3-0.6B-GGUF"
MODEL_FILE = "*.gguf"
MAX_NEW_TOKENS = 128
N_CTX = 1024


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

		response = self._llm.create_chat_completion(
			messages=[
				{
					"role": "system",
					"content": (
						"/no_think\n"
						"Using only the context below, answer the question concisely. "
						"2-3 sentences or less is ideal.\n\n"
						f"Context:\n{context}"
					),
				},
				{"role": "user", "content": question},
			],
			max_tokens=MAX_NEW_TOKENS,
			temperature=0.0,
		)
		answer = response["choices"][0]["message"]["content"]
		if "</think>" in answer:
			answer = answer.split("</think>", 1)[1]
		return answer.strip()
