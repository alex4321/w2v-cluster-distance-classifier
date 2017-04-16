from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from .cosine_distance import cosine_distance
import numpy
import pickle


class Classifier:
    def __init__(self, scaler, knn, maximal_distance):
        """
        Initialize classifier
        :param scaler: scaler
        :type scaler: MinMaxScaler
        :param knn: k-nearest neighbors classifier
        :type knn: KNeighborsClassifier
        :param maximal_distance: if distance to neighbours is bigger - replace their label confidence to zero
        :type maximal_distance: float
        """
        self.scaler = scaler
        self.knn = knn
        self.maximal_distance = maximal_distance

    def predict(self, text_vectors):
        values = []
        for vector in text_vectors:
            distances_tbl, indices_tbl = self.knn.kneighbors(self.scaler.transform([vector]))
            distances, indices = distances_tbl[0], indices_tbl[0]
            labels = self.knn._y[indices]
            k = numpy.ones([len(indices)])
            k[distances >= self.maximal_distance] = 0
            result_labels = (labels * k[:,None]).mean(axis=0)
            values.append(result_labels)
        values_np = numpy.array(values)
        std = values_np.std(axis=1)
        return values_np[std.argmax()]

    def dumps(self):
        X = self.knn._fit_X
        Y = self.knn._y
        return pickle.dumps([X, Y, self.scaler, self.maximal_distance])

    @staticmethod
    def loads(source):
        from .classifier_builder import ClassifierBuilder
        X, Y, scaler, maximal_distance = pickle.loads(source)
        knn = KNeighborsClassifier(n_neighbors=ClassifierBuilder.NEIGHBOURS_COUNT,
                                   metric='pyfunc',
                                   metric_params={
                                       'func': cosine_distance
                                   })
        knn.fit(X,Y)
        return Classifier(scaler, knn, maximal_distance)