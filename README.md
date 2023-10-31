### BeatriceVec

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Github](https://img.shields.io/pypi/v/BeatriceVec.svg)](https://github.com/foscraft/beatrice-project/tree/main/dist)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/BeatriceVec.svg)](https://github.com/foscraft/beatrice-project/tree/main/dist)

<div align="center">
<img src="media/BEATRICEVECTOR.svg" alt="BeatriceVec Logo" width="200px">
</div>
BeatriceVec is a powerful Python package/tool designed for generating word embeddings in the dimension of 600, without relying on any third-party packages. Word embeddings are vector representations of words that capture semantic relationships and meaning in a numerical format, enabling various natural language processing (NLP) tasks such as word similarity, text classification, and information retrieval.

With BeatriceVec, users can transform textual data into meaningful vector representations. These embeddings can capture semantic relationships between words, enabling algorithms and models to understand context and similarities between different words. This capability proves particularly useful in tasks such as sentiment analysis, language translation, and recommendation systems.

Create your embeddings with BeatriceVec and use them to query your model locally without using the internet.

>It utilizes a dimensionality of 600, providing a rich representation space that can capture nuanced semantic information. By incorporating a higher dimensionality, the embeddings can potentially encode more complex relationships and capture finer-grained distinctions between words, leading to improved performance in downstream NLP tasks.

>The package offers a user-friendly interface and straightforward API, making it accessible for both beginners and experienced practitioners. It provides functions to train custom word embeddings on user-specific text corpora, allowing users to fine-tune embeddings according to their specific domain or application requirements.

>It empowers developers and researchers to explore the world of word embeddings and leverage the power of contextual word representations in their NLP projects. Its self-contained implementation, high-dimensional embeddings, and ease of use make it a valuable tool for tasks such as text analysis, information retrieval, and language understanding.

>Overall, BeatriceVec is a reliable and efficient Python package for generating word embeddings, offering flexibility, performance, and ease of use to enhance various NLP applications and empower developers in the field of natural language processing.

#### Installation

Install package or wheel, both are found in `dist` folder

```bash
#WHEEL
pip install beatricevec-1.0.1-py3-none-any.whl
```

```bash
#PACKAGE
pip install beatricevec-1.0.1.tar.gz
```

Download the wheel or package [here](https://github.com/foscraft/beatrice-project/tree/main/dist)

#### Usage

```python
from beatricevec import BeatriceVec

corpus = ["I am learning", "Natural language processing", "with BeatriceVec"]
embedder = BeatriceVec(corpus)
embedder.build_vocab()
embedder.initialize_word_vectors()
embedder.train()

embeddings = embedder.get_embeddings()

for embedding in embeddings:
    print(embedding)
```

#### Documentation

##### Methods

- `build_vocab()`: Builds the vocabulary from the corpus.
- `initialize_word_vectors()`: Initializes the word vectors with random values.
- `train()`: Trains the embedding model using the Word2Vec algorithm.
- `update_vector(vector: list, context_vector: list)`: Updates the target word vector using gradient descent.
- `get_embeddings() -> list`: Retrieves the embeddings for all words in the vocabulary.
- `get_embedding(word: str) -> list`: Retrieves the embedding vector for a given word.


#### License

BeatriceVec is released under the [Apache2.0 License](https://opensource.org/license/apache-2-0/).

How to [CONTRIBUTE](https://opensource.o)
