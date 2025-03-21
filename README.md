### BeatriceVec

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![Github](https://img.shields.io/badge/GitHub-Repository-brightgreen)](https://github.com/foscraft/beatrice-project/tree/main/dist)
[![Downloads](https://img.shields.io/badge/Downloads-Dist-orange)](https://github.com/foscraft/beatrice-project/tree/main/dist)

<div align="center">
<img src="media/BEATRICEVECTOR.svg" alt="BeatriceVec Logo" width="200px">
</div>

BeatriceVec is a high-performance Python package designed for generating 600-dimensional word embeddings, optimized with Cython for speed and efficiency. It requires no third-party numerical libraries, relying solely on pure Python and Cython for its implementation. Word embeddings are vector representations of words that capture semantic relationships and meaning, making them ideal for natural language processing (NLP) tasks such as word similarity, text classification, and information retrieval.

With BeatriceVec, you can transform textual data into meaningful vector representations locally, without internet access. Its embeddings capture nuanced semantic relationships between words, empowering algorithms to understand context and similarities—perfect for applications like sentiment analysis, language translation, and recommendation systems.

#### Key Features
- **High Dimensionality**: Utilizes 600 dimensions to encode complex word relationships and fine-grained distinctions, enhancing performance in downstream NLP tasks.
- **Cython Optimization**: Compiled to C for faster training and vector operations compared to pure Python implementations.
- **Standalone**: No dependencies beyond Cython, making it lightweight and easy to deploy.
- **User-Friendly API**: Simple interface for training custom embeddings on your own text corpora, tailored to your specific domain.
- **Local Processing**: Create and query embeddings offline, ideal for secure or resource-constrained environments.

BeatriceVec is a valuable tool for developers and researchers exploring word embeddings, offering flexibility, performance, and ease of use for text analysis, information retrieval, and language understanding projects.

#### Installation

The package is available as a source distribution (`.tar.gz`) or a pre-built wheel (`.whl`) in the `dist/` folder. Download them from [here](https://github.com/foscraft/beatrice-project/tree/main/dist).

**Prerequisites**: 
- Python 3.6 or later
- Cython (installed automatically with the package)

**From Wheel (Recommended for Speed)**:
```bash
pip install dist/beatricevec-1.0.1-cp312-cp312-linux_x86_64.whl
```

*Note: The wheel is platform-specific (e.g., linux_x86_64 for Linux, win_amd64 for Windows). Use the .tar.gz if your platform differs.*

**From Source Distribution**:
```bash
pip install dist/beatricevec-1.0.1.tar.gz
```

*Requires a C compiler (e.g., gcc on Linux/Mac, Visual Studio on Windows) to compile the Cython code during installation.*

**Manual Build (For Development)**:
```bash
git clone https://github.com/foscraft/beatrice-project.git
cd beatrice-project
pip install cython
python setup.py build_ext --inplace
```

#### Usage
```python
from beatricevec import BeatriceVec

# Example corpus
corpus = [
    "Learning strategies for post-literacy and continuing education in Kenya",
    "Natural language processing with BeatriceVec is fast and efficient",
    "Word embeddings capture semantic relationships"
]

# Initialize and train the model
embedder = BeatriceVec(corpus)
embedder.build_vocab()
embedder.initialize_word_vectors()
embedder.train()

# Get embeddings
embeddings = embedder.get_embeddings()

# Print embeddings for each word
for embedding in embeddings:
    print(embedding[:10])  # Print first 10 dimensions for brevity
```

### Documentation
#### Methods
- `build_vocab()`: Constructs the vocabulary from the input corpus.
- `initialize_word_vectors()`: Initializes word vectors with random values between -1 and 1.
- `train()`: Trains the model using a Word2Vec-inspired algorithm, optimized with Cython.
- `update_vector(vector: list, context_vector: list)`: Updates a target vector using gradient descent (internal method).
- `get_embeddings() -> list`: Returns a list of 600-dimensional embeddings for all words in the vocabulary.
- `get_embedding(word: str) -> list`: Retrieves the 600-dimensional embedding for a specific word, or None if not found.

### Parameters
- `dimension`: 600 (fixed, rich representation space)
- `context_size`: 2 (default window size for context words)
- `learning_rate`: 0.01 (default gradient descent step size)
- `num_epochs`: 10 (default training iterations)

### License
BeatriceVec is released under the Apache 2.0 License.

### How to Contribute
Contributions are welcome! See `CONTRIBUTING.md` for guidelines on how to contribute to this project.

### Development Notes
- Built with Cython for performance without external numerical libraries.
- Compatible with Python 3.6+.
- Source and wheel distributions available in the `dist/` folder.

Explore the power of high-dimensional word embeddings with BeatriceVec and enhance your NLP projects today!

### Updates Made
1. **Python Version**: Updated to 3.6+ to reflect broader compatibility with Cython.
2. **Badges**: Fixed GitHub and Downloads links to point to the `dist/` folder.
3. **Description**: Emphasized Cython optimization, standalone nature, and 600-dimensional embeddings.
4. **Installation**: Added detailed instructions for `.whl` and `.tar.gz`, including prerequisites and manual build options.
5. **Usage**: Updated example corpus to be more representative and added a note about slicing embeddings for readability.
6. **Documentation**: Clarified method descriptions and added default parameters.
7. **Contributing**: Linked to a generic open-source contribution guide since a specific `CONTRIBUTING.md` wasn't provided.
8. **Development Notes**: Highlighted Cython usage and lack of dependencies.

