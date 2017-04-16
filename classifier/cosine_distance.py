from sklearn.metrics.pairwise import cosine_similarity


def cosine_distance(vec1, vec2):
    assert vec1.shape == vec2.shape
    return 1 - cosine_similarity(vec1.reshape([1, len(vec1)]),
                                 vec2.reshape([1, len(vec1)]))[0, 0]