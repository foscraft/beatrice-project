from beatricevec import BeatriceVec

corpus = [
    "Learning strategies for post-literacy and continuing education in Kenya, Nigeria, Tanzania, and the United Kingdom",
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
