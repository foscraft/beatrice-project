import random
from typing import Any, List, Dict

class BeatriceVec:
    """Custom implementation of FastText-like word embeddings."""

    def __init__(self, corpus: List[str], ngram_size: int = 3):  # type: ignore
        """
        Initializes the BeatriceVec embedding model.

        Args:
            corpus (list): List of sentences used for training the model.
            ngram_size (int): The size of character n-grams to be used.
        """
        self.corpus = corpus
        self.ngram_size = ngram_size
        self.word_vocab = []
        self.ngram_vocab = {}
        self.word2id = {}
        self.id2word = {}
        self.ngram2id = {}
        self.word_vectors = None
        self.ngram_vectors = None
        self.dimension = 600
        self.context_size = 2
        self.learning_rate = 0.01
        self.num_epochs = 10

    def build_vocab(self) -> None:
        """
        Builds the vocabulary of words and n-grams from the corpus.
        """
        for sentence in self.corpus:
            for word in sentence.split():
                if word not in self.word_vocab:
                    self.word_vocab.append(word)
                ngrams = self.generate_ngrams(word)
                for ngram in ngrams:
                    if ngram not in self.ngram_vocab:
                        self.ngram_vocab[ngram] = 1
                    else:
                        self.ngram_vocab[ngram] += 1

        self.word_vocab.sort()
        for i, word in enumerate(self.word_vocab):
            self.word2id[word] = i
            self.id2word[i] = word

        self.ngram_vocab = sorted(self.ngram_vocab.keys())
        for i, ngram in enumerate(self.ngram_vocab):
            self.ngram2id[ngram] = i

    def generate_ngrams(self, word: str) -> List[str]:
        """
        Generates character n-grams from a given word.

        Args:
            word (str): The word to generate n-grams from.

        Returns:
            list: List of n-grams.
        """
        ngrams = []
        word = f"<{word}>"
        for i in range(len(word) - self.ngram_size + 1):
            ngrams.append(word[i:i + self.ngram_size])
        return ngrams

    def initialize_vectors(self) -> None:
        """
        Initializes the word and n-gram vectors with random values.
        """
        vocab_size = len(self.word_vocab)
        ngram_vocab_size = len(self.ngram_vocab)
        self.word_vectors = [
            [random.uniform(-1, 1) for _ in range(self.dimension)]
            for _ in range(vocab_size)
        ]
        self.ngram_vectors = [
            [random.uniform(-1, 1) for _ in range(self.dimension)]
            for _ in range(ngram_vocab_size)
        ]

    def train(self) -> None:
        """
        Trains the embedding model using a simplified FastText-like algorithm.
        """
        for _ in range(self.num_epochs):
            for sentence in self.corpus:
                tokens = sentence.split()
                for i, target_word in enumerate(tokens):
                    target_index = self.word2id[target_word]
                    context_start = max(0, i - self.context_size)
                    context_end = min(i + self.context_size + 1, len(tokens))
                    context_words = (
                        tokens[context_start:i] + tokens[i + 1 : context_end]
                    )

                    target_vector = self.word_vectors[target_index]

                    for context_word in context_words:
                        context_index = self.word2id[context_word]
                        context_vector = self.word_vectors[context_index]

                        # Update word vectors using gradient descent
                        self.update_vector(target_vector, context_vector)
                        self.update_vector(context_vector, target_vector)

                        # Update n-gram vectors
                        target_ngrams = self.generate_ngrams(target_word)
                        context_ngrams = self.generate_ngrams(context_word)

                        for ngram in target_ngrams:
                            ngram_index = self.ngram2id[ngram]
                            ngram_vector = self.ngram_vectors[ngram_index]
                            self.update_vector(target_vector, ngram_vector)

                        for ngram in context_ngrams:
                            ngram_index = self.ngram2id[ngram]
                            ngram_vector = self.ngram_vectors[ngram_index]
                            self.update_vector(context_vector, ngram_vector)

    def update_vector(self, vector: List[float], context_vector: List[float]) -> None:
        """
        Updates the target word vector using gradient descent.

        Args:
            vector (list): The target word vector.
            context_vector (list): The context word vector.
        """
        for i in range(self.dimension):
            vector[i] -= self.learning_rate * context_vector[i]

    def get_embeddings(self) -> Dict[str, List[float]]:
        """
        Retrieves the embeddings for all words in the vocabulary.

        Returns:
            dict: Dictionary of words and their embedding vectors.
        """
        return {word: self.get_embedding(word) for word in self.word_vocab}

    def get_embedding(self, word: str) -> List[float]:
        """
        Retrieves the embedding vector for a given word.

        Args:
            word (str): The word to get the embedding for.

        Returns:
            list: The embedding vector for the given word.
        """
        if word in self.word2id:
            word_vector = self.word_vectors[self.word2id[word]]
            ngram_vectors = [self.ngram_vectors[self.ngram2id[ngram]] for ngram in self.generate_ngrams(word) if ngram in self.ngram2id]
            combined_vector = word_vector[:]
            for ngram_vector in ngram_vectors:
                combined_vector = [sum(x) for x in zip(combined_vector, ngram_vector)]
            return combined_vector
        return [0.0] * self.dimension
