import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
from tensorflow.keras.utils import to_categorical
from .lib import *
from sklearn.preprocessing import MinMaxScaler
import nltk

class Vectorizer:
    __vectorizer = None
    __matrice = None

    def __init__(self, docs_list=None, vectorizer_type='count', ngram_tuple=(1,1), space_dimension=None, dataframe=None):
        if docs_list is not None:
            if vectorizer_type == 'count':
                cv = CountVectorizer(ngram_range=ngram_tuple)
                matrice = cv.fit_transform(docs_list)
                self.__vectorizer = cv
                self.__matrice = matrice
            elif vectorizer_type == 'tfidf':
                tfidfv = TfidfVectorizer(max_features=space_dimension, stop_words=load_stop_words('english'), preprocessor=self.preprocessor)
                matrice = tfidfv.fit_transform(docs_list)
                self.__vectorizer = tfidfv
                self.__matrice = matrice
            elif vectorizer_type == 'custom':
                features = np.vectorize(Vectorizer.get_custom_features)
                data = features(docs_list)
                v = DictVectorizer()
                matrice = v.fit_transform(data)
                self.__vectorizer = v
                self.__matrice = matrice
            elif vectorizer_type == 'min_max':
                self.__vectorizer = MinMaxScaler()
                self.__matrice = self.__vectorizer.fit_transform(dataframe)
            else:
                pass

    def get_sparse_matrix(self):
        return self.__matrice
    
    def get_matrix(self):
        return self.__matrice.toarray()

    def get_vectorizer(self):
        return self.__vectorizer

    def get_features_names(self):
        return self.__vectorizer.get_feature_names()

    @staticmethod
    def tokenizer(doc):
        return doc.split()

    @staticmethod
    def preprocessor(doc):
        """def my_tokenizer(s):
        return s.split()
        vectorizer = CountVectorizer(tokenizer=my_tokenizer)

        """
        tokens = doc.split(' ')
        result = []
        for p in tokens:
            result.append(nltk.stem.PorterStemmer().stem(p))
        return ' '.join(result)

    @staticmethod
    def get_custom_features(e):
        e = e.lower()
        return {
            'f1': e[0],  # First letter
            'f2': e[0:2],  # First 2 letters
            'f3': e[0:3],  # First 3 letters
            'l1': e[-1],
            'l2': e[-2:],
            'l3': e[-3:],
        }

    def get_docs_projections_as_sparse(self, docs_liste, projection_type='normal'):
        if projection_type != 'normal':
            docs_liste = np.vectorize(Vectorizer.get_custom_features)(docs_liste)
        return self.__vectorizer.transform(docs_liste)

    def save_vectorizer(self, vectorizer_path='data/vectorizer.data'):
        out_vectorizer_file = open(vectorizer_path, 'wb')
        joblib.dump(self.__vectorizer, out_vectorizer_file)
        out_vectorizer_file.close()

    def load_vectorizer(self, vectorizer_path='data/vectorizer.data'):
        self.__vectorizer = joblib.load(open(vectorizer_path, 'rb'))

    def reshape(self, new_shpae_tuple):
        self.__matrice = np.array(self.__matrice.reshape(new_shpae_tuple))

    def get_sum_by_columns_as_list(self):
        count_list = np.array(self.get_sparse_matrix().sum(axis=0))
        count_list = count_list.reshape(self.get_shape()[1])
        return count_list

    def get_sum_by_rows_as_list(self):
        count_list = np.array(self.get_sparse_matrix().sum(axis=1))
        count_list = count_list.reshape(self.get_shape()[0])
        return count_list

    def get_shape(self):
        return self.__matrice.shape

    @staticmethod
    def to_one_hot(vecteur_of_categories):
        """converti une colone avec des categorie mais numerique en forme One Hot Encoding exemple versicolor
        est de label 2 se transform en [0 0 1]"""
        return to_categorical(vecteur_of_categories)

    @staticmethod
    def get_reshaped_matrix(matrix, new_shape_tuple):
        print(new_shape_tuple)
        new_matrix = matrix.reshape(new_shape_tuple)
        print('okkkk {}'.format(new_matrix.shape))
        return new_matrix

    @staticmethod
    def reshape_images_for_cnn(images_as_liste):
        images_as_liste.reshape(images_as_liste.shape[0], images_as_liste.shape[1], images_as_liste.shape[1], 1) \
            .astype('float32')

from nltk.stem.isri import ISRIStemmer


def stemming(string_in):
    tokens = string_in.split()
    new_tokens = []
    stemmer = ArabicLightStemmer()
    for p in tokens:
        new_tokens.append(stemmer.light_stem(p))
    return ''.join(new_tokens)

