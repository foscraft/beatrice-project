import pytest

from beatricevec.main import (  # Import your BeatriceVec class from the main code
    BeatriceVec,
)

# Create some sample sentences for testing
sample_corpus = [
    "Learning strategies for post-literacy and continuing education in Kenya, Nigeria, Tanzania, and the United Kingdom",
    "Family planning and sexual behavior in the era of HIVAIDS the case of Nakuru District Kenya.. Recently the prevalence of contraceptive use has increased in Kenya.",
    "ECHIDNOPSIS GLOBOSA SPNOV ASCLEPIADACEAESTAPELIEAE FROM YEMEN. The new species Echidnopsis globosa from rocky hillsides on Limestone in the Hadramaut Region in Yemen is described and illustrated.",
]

@pytest.fixture
def beatrice_vec_instance():
    return BeatriceVec(sample_corpus)

def test_initialize(beatrice_vec_instance):
    assert beatrice_vec_instance.corpus == sample_corpus
    assert beatrice_vec_instance.word_vocab == []
    assert beatrice_vec_instance.word2id == {}
    assert beatrice_vec_instance.id2word == {}
    assert beatrice_vec_instance.word_vectors is None
    assert beatrice_vec_instance.dimension == 600
    assert beatrice_vec_instance.context_size == 2
    assert beatrice_vec_instance.learning_rate == 0.01
    assert beatrice_vec_instance.num_epochs == 10


def test_build_vocab(beatrice_vec_instance):
    beatrice_vec_instance.build_vocab()
    assert len(beatrice_vec_instance.word_vocab) > 0
    assert len(beatrice_vec_instance.word_vocab) == len(beatrice_vec_instance.word2id)
    assert len(beatrice_vec_instance.word_vocab) == len(beatrice_vec_instance.id2word)


def test_initialize_word_vectors(beatrice_vec_instance):
    beatrice_vec_instance.build_vocab()
    beatrice_vec_instance.initialize_word_vectors()
    assert beatrice_vec_instance.word_vectors is not None


def test_update_vector(beatrice_vec_instance):
    beatrice_vec_instance.build_vocab()
    beatrice_vec_instance.initialize_word_vectors()
    vector = beatrice_vec_instance.word_vectors[0]
    context_vector = beatrice_vec_instance.word_vectors[1]
    original_vector = vector.copy()
    beatrice_vec_instance.update_vector(vector, context_vector)
    assert vector != original_vector


def test_train(beatrice_vec_instance):
    beatrice_vec_instance.build_vocab()
    beatrice_vec_instance.initialize_word_vectors()
    beatrice_vec_instance.train()
    assert beatrice_vec_instance.word_vectors is not None


def test_get_embedding(beatrice_vec_instance):
    beatrice_vec_instance.build_vocab()
    beatrice_vec_instance.initialize_word_vectors()
    word = beatrice_vec_instance.word_vocab[0]
    embedding = beatrice_vec_instance.get_embedding(word)
    assert embedding is not None


def test_get_embeddings(beatrice_vec_instance):
    beatrice_vec_instance.build_vocab()
    beatrice_vec_instance.initialize_word_vectors()
    embeddings = beatrice_vec_instance.get_embeddings()
    assert len(embeddings) == len(beatrice_vec_instance.word_vocab)


if __name__ == "__main__":
    pytest.main()
