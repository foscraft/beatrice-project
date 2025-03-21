cimport cython
from libc.stdlib cimport rand, RAND_MAX

# Define float type
ctypedef double DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class BeatriceVec:
    cdef public list corpus, word_vocab
    cdef public dict word2id, id2word
    cdef public list word_vectors  # Changed from np.ndarray to list
    cdef public int dimension, context_size, num_epochs
    cdef public double learning_rate

    def __init__(self, corpus):
        self.corpus = corpus
        self.word_vocab = []
        self.word2id = {}
        self.id2word = {}
        self.word_vectors = None
        self.dimension = 600
        self.context_size = 2
        self.learning_rate = 0.01
        self.num_epochs = 10

    cpdef void build_vocab(self):
        cdef str sentence, word
        cdef int i
        
        for sentence in self.corpus:
            for word in sentence.split():
                if word not in self.word2id:
                    self.word_vocab.append(word)
        
        self.word_vocab.sort()
        for i, word in enumerate(self.word_vocab):
            self.word2id[word] = i
            self.id2word[i] = word

    cpdef void initialize_word_vectors(self):
        cdef int vocab_size = len(self.word_vocab)
        cdef int i, j
        cdef list vector
        
        # Initialize word_vectors as a list of lists
        self.word_vectors = []
        for i in range(vocab_size):
            vector = []
            for j in range(self.dimension):
                # Generate random number between -1 and 1
                vector.append(-1.0 + 2.0 * (<double>rand() / RAND_MAX))
            self.word_vectors.append(vector)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void train(self):
        cdef int epoch, i, target_index, context_index
        cdef int context_start, context_end
        cdef list tokens, context_words
        cdef str sentence, target_word, context_word
        cdef list target_vector, context_vector
        
        for epoch in range(self.num_epochs):
            for sentence in self.corpus:
                tokens = sentence.split()
                for i, target_word in enumerate(tokens):
                    target_index = self.word2id[target_word]
                    
                    context_start = max(0, i - self.context_size)
                    context_end = min(i + self.context_size + 1, len(tokens))
                    context_words = tokens[context_start:i] + tokens[i + 1:context_end]
                    
                    target_vector = self.word_vectors[target_index]
                    for context_word in context_words:
                        context_index = self.word2id[context_word]
                        context_vector = self.word_vectors[context_index]
                        
                        self.update_vector(target_vector, context_vector)
                        self.update_vector(context_vector, target_vector)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_vector(self, list vector, list context_vector):
        cdef int i
        cdef double lr = self.learning_rate
        cdef double val
        
        for i in range(self.dimension):
            val = <double>vector[i]
            val -= lr * <double>context_vector[i]
            vector[i] = val

    cpdef list get_embeddings(self):
        cdef list embeddings = []
        cdef str word
        for word in self.word_vocab:
            embedding = self.get_embedding(word)
            if embedding is not None:
                embeddings.append(embedding)
        return embeddings

    cpdef list get_embedding(self, str word):
        if word in self.word2id:
            return self.word_vectors[self.word2id[word]]
        return None