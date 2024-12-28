#!/usr/bin/env python3

from typing import List

import gensim.downloader as api
from gensim.models import KeyedVectors


def _load_model(model_path: str) -> KeyedVectors:
    """
    Load a pre-trained Word2Vec model in binary format.

    Args:
        model_path (str): Path to the Word2Vec model file.

    Returns:
        KeyedVectors: Loaded Word2Vec model.
    """
    return KeyedVectors.load_word2vec_format(model_path, binary=True)


def _combine_word_vectors(
        model: KeyedVectors, word1: str, word2: str, method: str = "add"
) -> List[float]:
    """
    Combine the vector values of two words.

    Args:
        model (KeyedVectors): The Word2Vec model.
        word1 (str): The first word.
        word2 (str): The second word.
        method (str): The method for combining vectors ('add' or 'average').

    Returns:
        List[float]: The resulting composite vector.
    """
    if word1 not in model or word2 not in model:
        raise ValueError(f"One or both words not in the model vocabulary: {word1}, {word2}")

    vec1 = model[word1]
    vec2 = model[word2]

    if method == "add":
        return vec1 + vec2
    elif method == "average":
        return (vec1 + vec2) / 2
    else:
        raise ValueError("Invalid method. Choose 'add' or 'average'.")


def _find_similar_words_for_vector(model: KeyedVectors, composite_vector: List[float], topn: int = 10):
    """
    Find words similar to a composite vector.

    Args:
        model (KeyedVectors): The Word2Vec model.
        composite_vector (List[float]): The composite vector.
        topn (int): Number of similar words to retrieve.

    Returns:
        List[tuple]: List of similar words with their similarity scores.
    """
    return model.similar_by_vector(composite_vector, topn=topn)


def combine_words_and_explore_neighbors(first_word: str, second_word: str) -> [str]:
    model_path = "GoogleNews-vectors-negative300.bin"
    model = _load_model(model_path)

    if not first_word:
        first_word = input("Enter the first word: ")
    if not second_word:
        second_word = input("Enter the second word: ")

    try:
        composite_vector = _combine_word_vectors(model, first_word, second_word, method="average")
        similar_words = _find_similar_words_for_vector(model, composite_vector)
        print("\nWords similar to the composite vector:")
        for word, similarity in similar_words:
            print(f"{word}: {similarity:.4f}")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    words = combine_words_and_explore_neighbors("honor", "odium")
    print("1:\n")
    print(words)
    print("\n\n 2:\n")
    words = combine_words_and_explore_neighbors("preservation", "ruin")
    print(words)
