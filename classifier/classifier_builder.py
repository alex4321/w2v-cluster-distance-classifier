import itertools
import numpy
from .text_vector_builder import W2vPosClusterTextVectorBuilder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from .classifier import Classifier
from .cosine_distance import cosine_distance
from .text_classifier import TextClassifier


class ClassifierBuilder:
    NEIGHBOURS_COUNT = 5

    def __init__(self, word2vec, global_clusters, morph_analyzer, dataset):
        self.word2vec = word2vec
        self.global_clusters = global_clusters
        self.morph_analyzer = morph_analyzer
        self.dataset = dataset
        vector_builder = W2vPosClusterTextVectorBuilder(word2vec, global_clusters, morph_analyzer)
        self.global_vectors, self.labels = vector_builder.process_dataset(dataset)
        self.unique_labels = list(set(itertools.chain(*self.labels)))
        self.unique_labels.sort()

    def labels_numbers(self):
        return numpy.array([
            [int(label in row) for label in self.unique_labels]
            for row in self.labels
        ])

    def _label_vectors(self, label):
        indices = numpy.array([int(label in row) for row in self.labels]) == 1
        vectors = self.global_vectors[indices]
        return vectors

    def label_top_clusters(self, label, top_clusters):
        mean = self._label_vectors(label).mean(axis=0)
        return mean.argsort()[::-1][:top_clusters]

    def labels_top_clusters(self, top_clusters):
        result = {}
        for label in self.unique_labels:
            result[label] = self.label_top_clusters(label, top_clusters)
        return result

    def cluster_words(self, cluster_index):
        return [row[0] for row in self.word2vec.most_similar([self.global_clusters[cluster_index]])]

    def class_mean_metric(self):
        vectors = [self._label_vectors(label).mean(axis=0)
                   for label in self.unique_labels]
        values = []
        for i, vector in enumerate(vectors):
            for j, other_vector in enumerate(vectors):
                if i == j:
                    continue
                values.append(cosine_distance(vector, other_vector))
        return float(numpy.array(values).mean())

    def create_classifier(self, use_clusters, maximal_distance):
        local_clusters = self.global_clusters[use_clusters]
        scaler = MinMaxScaler()
        vector_builder = W2vPosClusterTextVectorBuilder(self.word2vec, local_clusters, self.morph_analyzer)
        local_vectors_nonnormalized, _ = vector_builder.process_dataset(self.dataset)
        local_vectors = scaler.fit_transform(local_vectors_nonnormalized)
        knn = KNeighborsClassifier(n_neighbors=ClassifierBuilder.NEIGHBOURS_COUNT,
                                   metric='pyfunc',
                                   metric_params={
                                       'func': cosine_distance
                                   })
        knn.fit(local_vectors, self.labels_numbers())
        classifier = Classifier(scaler, knn, maximal_distance)
        vectorizer = W2vPosClusterTextVectorBuilder(self.word2vec, local_clusters, self.morph_analyzer)
        return TextClassifier(classifier, vectorizer, self.unique_labels)