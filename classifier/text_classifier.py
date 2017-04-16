import pickle
from .classifier import Classifier
from .text_vector_builder import W2vPosClusterTextVectorBuilder


class TextClassifier:
    def __init__(self, backend, vectorizer, labels):
        self.backend = backend
        self.vectorizer = vectorizer
        self.labels = labels

    def predict(self, text):
        confidences = self.backend.predict(self.vectorizer.build_text_vectors(text))
        result = {}
        for i, label in enumerate(self.labels):
            result[label] = confidences[i]
        return result

    def dumps(self):
        return pickle.dumps([self.backend.dumps(), self.vectorizer.clusters, self.labels])

    @staticmethod
    def loads(dump, word2vec, morph_analyzer):
        backend_dump, vectorizer_clusters, labels = pickle.loads(dump)
        vectorizer = W2vPosClusterTextVectorBuilder(word2vec, vectorizer_clusters, morph_analyzer)
        return TextClassifier(Classifier.loads(backend_dump), vectorizer, labels)