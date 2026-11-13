from collections.abc import Sequence

from .text import normalize_text


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length")
    errors = 0
    word_count = 0
    for reference, hypothesis in zip(references, hypotheses):
        reference_words = normalize_text(reference).split()
        hypothesis_words = normalize_text(hypothesis).split()
        errors += _edit_distance(reference_words, hypothesis_words)
        word_count += len(reference_words)
    return errors / max(word_count, 1)


def character_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length")
    errors = 0
    character_count = 0
    for reference, hypothesis in zip(references, hypotheses):
        reference_text = normalize_text(reference)
        hypothesis_text = normalize_text(hypothesis)
        errors += _edit_distance(reference_text, hypothesis_text)
        character_count += len(reference_text)
    return errors / max(character_count, 1)
