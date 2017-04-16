import json
from gensim.models import KeyedVectors
from nltk import word_tokenize
import numpy
from pymorphy2 import MorphAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from .dataset_item import DatasetItem


class W2vPosClusterTextVectorBuilder:
    SIMILARITY_THRESHOLD = 0.4
    MORPH_MIN_SCORE = 0.2

    def __init__(self, word2vec, clusters, morph_analyzer):
        """
        Initialize
        :param word2vec: word2vec model. Note that each word must be written as word_POS (e.g. year_NOUN)
        :type word2vec: KeyedVectors
        :param clusters: clusters
        :type clusters: numpy.ndarray
        :param morph_analyzer: morphologycal analyzer
        :type morph_analyzer: MorphAnalyzer
        """
        self.word2vec = word2vec
        self.clusters = clusters
        self.morph_analyzer = morph_analyzer

    def _word_pos_combination(self, word, pos):
        if pos.startswith("ADJ"):
            pos = "ADJ"
        elif pos.startswith("ADV"):
            pos = "ADV"
        return word + "_" + pos

    def _morph_to_str(self, morph):
        return self._word_pos_combination(morph.normal_form, morph.tag.POS)

    def _use_morph(self, morph):
        if not morph.tag.POS:
            return False
        if morph.score < W2vPosClusterTextVectorBuilder.MORPH_MIN_SCORE:
            return False
        text = self._morph_to_str(morph)
        return text in self.word2vec

    def _combinations(self, items):
        if len(items) == 0:
            return [[]]
        first = items[0]
        others = self._combinations(items[1:])
        values = []
        for item in first:
            for other in others:
                values.append([item] + other)
            if len(others) == 0:
                values.append([item])
        if len(first) == 0:
            for other in others:
                values.append(other)
            if len(others) == 0:
                values.append([[]])
        return values

    def _uniqie(self, versions):
        json_versions = list(set([json.dumps(version, ensure_ascii=False) for version in versions]))
        json_versions.sort()
        return [json.loads(version, encoding="utf-8") for version in json_versions]

    def _parse_text(self, text):
        tokens = word_tokenize(text)
        morphs = [self.morph_analyzer.parse(token)
                  for token in tokens]
        versions = []
        for token_morphs in morphs:
            token_morphs_converted = []
            for morph in token_morphs:
                if not self._use_morph(morph):
                    continue
                token_morphs_converted.append(self._morph_to_str(morph))
            versions.append(token_morphs_converted)
        return self._uniqie(self._combinations(versions))

    def _token_vector(self, token):
        word2vec_vector = self.word2vec[token].reshape([1, self.word2vec.vector_size])
        similarities = cosine_similarity(word2vec_vector, self.clusters).reshape([len(self.clusters)])
        similarities[similarities < W2vPosClusterTextVectorBuilder.SIMILARITY_THRESHOLD] = 0
        return similarities

    def build_text_vectors(self, text):
        """
        Build text vector
        :param text: text
        :type text: str
        :return: vector versions
        :rtype: list[numpy.ndarray]
        """
        parse_versions = self._parse_text(text)
        vectors = []
        for parse_version in parse_versions:
            tokens_vectors = [numpy.zeros([len(self.clusters)])] +\
                             [self._token_vector(token) for token in parse_version]
            vectors.append(numpy.array(tokens_vectors).sum(axis=0))
        return vectors

    def process_dataset(self, dataset):
        """
        Process dataset
        :param dataset: dataset
        :type dataset: list[DatasetItem]
        :return: list of text vector/labels
        :rtype: (numpy.ndarray, list[list[str]])
        """
        flat_vectors = []
        flat_labels = []
        for item in dataset:
            vector_versions = self.build_text_vectors(item.text)
            for vector in vector_versions:
                flat_vectors.append(vector)
                flat_labels.append(item.labels)
        return numpy.array(flat_vectors), flat_labels