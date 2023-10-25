"""_summary_
This package was built by Reuben Nyaribari
Returns:
    _type_: _description_
"""
import random
from typing import Any


class BeatriceVec:
    """_summary_"""

    def __init__(self, corpus):  # type: ignore
        """
        Initializes the BeatriceVec embedding model.

        Args:
            corpus (list): List of sentences used for training the model.
        """
        self.corpus = corpus
        self.word_vocab = []
        self.word2id = {}
        self.id2word = {}
        self.word_vectors = None
        self.dimension = 600
        self.context_size = 2
        self.learning_rate = 0.01
        self.num_epochs = 10

    def build_vocab(self) -> None:
        """
        Builds the vocabulary from the corpus.
        """
        for sentence in self.corpus:
            for word in sentence.split():
                if word not in self.word_vocab:
                    self.word_vocab.append(word)

        self.word_vocab.sort()
        for i, word in enumerate(self.word_vocab):
            self.word2id[word] = i
            self.id2word[i] = word

    def initialize_word_vectors(self) -> None:
        """
        Initializes the word vectors with random values.
        """
        vocab_size = len(self.word_vocab)
        self.word_vectors = [
            [random.uniform(-1, 1) for _ in range(self.dimension)]
            for _ in range(vocab_size)
        ]

    def train(self) -> None:
        """
        Trains the embedding model using the Word2Vec algorithm.
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

                    for context_word in context_words:
                        context_index = self.word2id[context_word]

                        target_vector = self.word_vectors[target_index]  # type: ignore
                        context_vector = self.word_vectors[context_index]  # type: ignore

                        # Update word vectors using gradient descent
                        self.update_vector(target_vector, context_vector)
                        self.update_vector(context_vector, target_vector)

    def update_vector(self, vector: Any, context_vector: Any) -> Any:
        """
        Updates the target word vector using gradient descent.

        Args:
            vector (list): The target word vector.
            context_vector (list): The context word vector.
        """
        for i in range(self.dimension):
            vector[i] -= self.learning_rate * context_vector[i]

    def get_embeddings(self) -> Any:
        """
        Retrieves the embeddings for all words in the vocabulary.

        Returns:
            list: List of embedding vectors for each word in the vocabulary.
        """
        embeddings = []
        for word in self.word_vocab:
            embedding = self.get_embedding(word)
            if embedding is not None:
                embeddings.append(embedding)
        return embeddings

    def get_embedding(self, word: Any) -> Any:
        """
        Retrieves the embedding vector for a given word.

        Args:
            word (str): The word to get the embedding for.

        Returns:
            list: The embedding vector for the given word.
        """
        return self.word_vectors[self.word2id[word]] if word in self.word2id else None  # type: ignore


corpus = [
    "Stratgies dapprentissage pour la postalphabtisation et lducation continue au Kenya au Nigria en Tanzanie et au RoyaumeUni",
    "Family planning and sexual behavior in the era of HIVAIDS the case of Nakuru District Kenya.. Recently the prevalence of contraceptive use has increased in Kenya.",
    "ECHIDNOPSIS GLOBOSA SPNOV ASCLEPIADACEAESTAPELIEAE FROM YEMEN. The new species Echidnopsis globosa from rocky hillsides on Limestone in the Hadramaut Region in Yemen is described and illustrated.",
]
embedder = BeatriceVec(corpus)
embedder.build_vocab()
embedder.initialize_word_vectors()
embedder.train()

embeddings = embedder.get_embeddings()
lst = list(embeddings)
print(lst)