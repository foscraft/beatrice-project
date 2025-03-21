cimport cython # type: ignore
from libc.stdlib cimport rand, RAND_MAX # type: ignore

# Define float type
ctypedef double DTYPE_t # type: ignore

@cython.boundscheck(False) # type: ignore
@cython.wraparound(False) # type: ignore
cdef class BeatriceVec:
    cdef public list corpus, word_vocab # type: ignore
    cdef public dict word2id, id2word # type: ignore
    cdef public list word_vectors  # Changed from np.ndarray to list
    cdef public int dimension, context_size, num_epochs # type: ignore
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

    cpdef void build_vocab(self): # type: ignore
        cdef str sentence, word # type: ignore
        cdef int i # type: ignore
        
        for sentence in self.corpus: # type: ignore
            for word in sentence.split():
                if word not in self.word2id: # type: ignore
                    self.word_vocab.append(word) # type: ignore
        
        self.word_vocab.sort() # type: ignore
        for i, word in enumerate(self.word_vocab): # type: ignore
            self.word2id[word] = i # type: ignore
            self.id2word[i] = word # type: ignore

    cpdef void initialize_word_vectors(self): # type: ignore
        cdef int vocab_size = len(self.word_vocab) # type: ignore
        cdef int i, j # type: ignore
        cdef list vector # type: ignore
        
        # Initialize word_vectors as a list of lists
        self.word_vectors = [] # type: ignore
        for i in range(vocab_size):
            vector = []
            for j in range(self.dimension): # type: ignore
                # Generate random number between -1 and 1
                vector.append(-1.0 + 2.0 * (<double>rand() / RAND_MAX))
            self.word_vectors.append(vector) # type: ignore

    @cython.boundscheck(False) # type: ignore
    @cython.wraparound(False) # type: ignore
    cpdef void train(self):
        cdef int epoch, i, target_index, context_index # type: ignore
        cdef int context_start, context_end # type: ignore
        cdef list tokens, context_words # type: ignore
        cdef str sentence, target_word, context_word # type: ignore
        cdef list target_vector, context_vector # type: ignore
        
        for epoch in range(self.num_epochs): # type: ignore
            for sentence in self.corpus: # type: ignore
                tokens = sentence.split()
                for i, target_word in enumerate(tokens):
                    target_index = self.word2id[target_word] # type: ignore
                    
                    context_start = max(0, i - self.context_size) # type: ignore
                    context_end = min(i + self.context_size + 1, len(tokens)) # type: ignore
                    context_words = tokens[context_start:i] + tokens[i + 1:context_end]
                    
                    target_vector = self.word_vectors[target_index] # type: ignore
                    for context_word in context_words:
                        context_index = self.word2id[context_word] # type: ignore
                        context_vector = self.word_vectors[context_index] # type: ignore
                        
                        self.update_vector(target_vector, context_vector) # type: ignore
                        self.update_vector(context_vector, target_vector) # type: ignore

    @cython.boundscheck(False) # type: ignore
    @cython.wraparound(False) # type: ignore
    cdef void update_vector(self, list vector, list context_vector): # type: ignore
        cdef int i # type: ignore
        cdef double lr = self.learning_rate # type: ignore
        cdef double val # type: ignore
        
        for i in range(self.dimension): # type: ignore
            val = <double>vector[i]
            val -= lr * <double>context_vector[i]
            vector[i] = val

    cpdef list get_embeddings(self): # type: ignore
        cdef list embeddings = [] # type: ignore
        cdef str word # type: ignore
        for word in self.word_vocab: # type: ignore
            embedding = self.get_embedding(word) # type: ignore
            if embedding is not None:
                embeddings.append(embedding)
        return embeddings

    cpdef list get_embedding(self, str word):
        if word in self.word2id:
            return self.word_vectors[self.word2id[word]]
        return None