from typing import List 
from .models import AnsweredQuestion, MinimalSource , StudentSearchResults

def overlap_ratio(
	retrived: MinimalSource,
	correct: MinimalSource,
) -> float:
	overlap_start = max(retrived.first_character_index, correct.first_character_index)
	overlap_end = min(retrived.last_character_index, correct.last_character_index)
	overlap_length = max(0, overlap_end - overlap_start)
	correct_length = correct.last_character_index - correct.first_character_index
	if correct_length == 0:
		return 0.0
	return overlap_length / correct_length

def recall_at_k(
    student_results: StudentSearchResults,
    ground_truth: List[AnsweredQuestion],
    k: int,
    min_overlap: float = 0.05,
) -> float:
	question_map = {q.question_id: q.sources for q in ground_truth}
	scores: List[float] = []
	for result in student_results.search_results:
		correct_sources = question_map.get(result.question_id, [])
		if not correct_sources:
			continue
		retriveds = result.retrieved_sources[:k]
		found = 0
		for correct in correct_sources:
			for retrived in retriveds:
				if overlap_ratio(retrived, correct) >= min_overlap:
					found += 1
					break
		scores.append(found / len(correct_sources))
	return sum(scores) / len(scores) if scores else 0.0