from fire import Fire
from .indexer import build_index
from .retriever import Retriever
from .generator import Generator
from .evaluator import recall_at_k
from .models import *
from tqdm import tqdm
from pathlib import Path

class RAGSystem:
	def index(self, repo_path: str, max_chunk_size: int = 2000):
		build_index(repo_path=repo_path, max_chunk_size=max_chunk_size)
		
	def search(self, query: str, k: int = 10):
		retriever = Retriever()
		return retriever.search(query=query, k=k)

	def search_dataset(self, dataset_path: str, k: int = 10, save_directory: str = "data/output/search_results"):
		retriever = Retriever()
		dataset = RagDataset.model_validate_json(Path(dataset_path).read_text())
		results: List[MinimalSearchResults] = []
		for q in tqdm(dataset.rag_questions, desc="Searching"):
			sources = retriever.search(query=q.question, k=k)
			results.append(MinimalSearchResults(question_id=q.question_id, question_str=q.question, retrieved_sources=sources))
		out = StudentSearchResults(search_results=results, k=k)
		out_path = Path(save_directory)
		out_path.mkdir(parents=True, exist_ok=True)
		fname = Path(dataset_path).name
		(out_path / fname).write_text(out.model_dump_json(indent=2))
		print(f"Search results saved to {out_path / fname}")
  
	def answer(self, query: str, k: int = 10):
		retriever = Retriever()
		generator = Generator()
		sources = retriever.search(query=query, k=k)
		answer = generator.generate(question=query, sources=sources)
		return answer

	def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
		generator = Generator()
		data = StudentSearchResults.model_validate_json(
			Path(student_search_results_path).read_text()
		)
		answers: List[MinimalAnswer] = []

		for res in tqdm(data.search_results, desc="Generating answers"):
			ans = generator.generate(res.question_str, res.retrieved_sources)
			answers.append(
				MinimalAnswer(
					question_id=res.question_id,
					question_str=res.question_str,
					retrieved_sources=res.retrieved_sources,
					answer=ans,
				)
			)

		out = StudentSearchResultsAndAnswer(search_results=answers, k=data.k)
		out_path = Path(save_directory)
		out_path.mkdir(parents=True, exist_ok=True)
		fname = Path(student_search_results_path).name
		(out_path / fname).write_text(out.model_dump_json(indent=2))
		print(f"Saved student_search_results_and_answer to {out_path / fname}")
  
  
	def evaluate(
		self,
		student_answer_path: str,
		dataset_path: str,
		k: int = 10,
	) -> None:
		"""Evaluate search results against ground truth."""
		student = StudentSearchResults.model_validate_json(
			Path(student_answer_path).read_text()
		)
		gt_dataset = RagDataset.model_validate_json(Path(dataset_path).read_text())
		gt = [
			q for q in gt_dataset.rag_questions
			if isinstance(q, AnsweredQuestion)
		]

		print("Evaluation Results")
		print("=" * 40)
		print(f"Questions evaluated: {len(gt)}")
		for ki in [1, 3, 5, 10]:
			if ki <= k:
				score = recall_at_k(student, gt, k=ki)
				print(f"Recall@{ki}:  {score:.3f}")


if __name__ == "__main__":
	Fire(RAGSystem())