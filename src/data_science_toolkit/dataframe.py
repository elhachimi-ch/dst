from datetime import timedelta
from math import ceil
import pandas as pd
from pyparsing import col
import scipy.sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from zmq import has
from .lib import Lib
from .vectorizer import Vectorizer
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.compose import ColumnTransformer
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator as SG
from sklearn.datasets import load_iris, load_boston
from collections import Counter
from .chart import Chart
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


class DataFrame:
    """
    """
    __vectorizer = None
    __generator = None

    def __init__(self, data_link=None, columns_names_as_list=None, data_types_in_order=None, delimiter=',',
                 data_type='csv', has_header=True, line_index=None, skip_empty_line=False, sheet_name='Sheet1'):
        if data_link is not None:
            if data_type == 'csv':
                if has_header is True:
                    self.__dataframe = pd.read_csv(data_link, encoding='utf-8', delimiter=delimiter, 
                                               low_memory=False, error_bad_lines=False, skip_blank_lines=False)
                else:
                    self.__dataframe = pd.read_csv(data_link, encoding='utf-8', delimiter=delimiter, 
                                               low_memory=False, error_bad_lines=False, skip_blank_lines=False,
                                               header=None)
            elif data_type == 'json':
                self.__dataframe = pd.read_json(data_link, encoding='utf-8')
            elif data_type == 'xls':
                self.__dataframe = pd.read_excel(data_link, sheet_name=sheet_name)
            elif data_type == 'pkl':
                self.__dataframe = pd.read_pickle(data_link)
            elif data_type == 'dict':
                self.__dataframe = pd.DataFrame.from_dict(data_link)
            elif data_type == 'matrix':
                index_name = [i for i in range(len(data_link))]
                colums_name = [i for i in range(len(data_link[0]))]
                self.__dataframe = pd.DataFrame(data=data_link, index=index_name, columns=colums_name)
            elif data_type == 'list':
                y = data_link
                if (not isinstance(y, pd.core.series.Series or not isinstance(y, pd.core.frame.DataFrame))):
                    y = np.array(y)
                    y = np.reshape(y, (y.shape[0],))
                    y = pd.Series(y)
                self.__dataframe = pd.DataFrame()
                if columns_names_as_list is not None:
                    self.__dataframe[columns_names_as_list[0]] = y
                else:
                    self.__dataframe['0'] = y
                    
                
                """data = array([['','Col1','Col2'],['Row1',1,2],['Row2',3,4]])
                pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:]) """
            elif data_type == 'df':
                self.__dataframe = data_link
            types = {}
            if data_types_in_order is not None and columns_names_as_list is not None:
                self.__dataframe.columns = columns_names_as_list
                for i in range(len(columns_names_as_list)):
                    types[columns_names_as_list[i]] = data_types_in_order[i]
            elif columns_names_as_list is not None:
                self.__dataframe.columns = columns_names_as_list
                for p in columns_names_as_list:
                    types[p] = str

            self.__dataframe = self.get_dataframe().astype(types)

            if line_index is not None:
                self.__dataframe.index = line_index
        else:
            self.__dataframe = pd.DataFrame()
        
    def get_generator(self):
        return self.__generator
    
    def remove_stopwords(self, column, language_or_stopwords_list='english', in_place=True):
        if isinstance(language_or_stopwords_list, list) is True:
            stopwords = language_or_stopwords_list
        elif language_or_stopwords_list == 'arabic':
            stopwords = Lib.read_text_file_as_list('data/arabic_stopwords.csv')
        else:
            nltk.download('stopwords')
            stopwords = nltk.corpus.stopwords.words(language_or_stopwords_list)
        self.transform_column(column, column, DataFrame.remove_stopwords_lambda, in_place, stopwords)
        return self.__dataframe
    
    @staticmethod
    def remove_stopwords_lambda(document, stopwords_list):
        document = str.lower(document)
        stopwords = stopwords_list
        words = word_tokenize(document)
        clean_words = []
        for w in words:
            if w not in stopwords:
                clean_words.append(w)
        return ' '.join(clean_words)
    
    def add_random_series_column(self, column_name='random',min=0, max=100, distrubution_type='random', mean=0, sd=1):
        if distrubution_type == 'random':
            series = pd.Series(np.random.randint(min, max, self.get_shape()[0]))
        elif distrubution_type == 'standard_normal':
            series = pd.Series(np.random.standard_normal(self.get_shape()[0]))
        elif distrubution_type == 'normal':
            series = pd.Series(np.random.normal(mean, sd, self.get_shape()[0]))
        else:
            series = pd.Series(np.random.randn(self.get_shape()[0]))
        self.add_column(series, column_name)
        return self.__dataframe
    
    def drop_full_nan_columns(self):
        for c in self.__dataframe.columns:
                miss = self.__dataframe[c].isnull().sum()
                missing_data_percent = round((miss/self.get_shape()[0])*100, 2)
                if missing_data_percent == 100:
                    self.drop_column(c)
                    
    def drop_columns_with_nan_threshold(self, threshold=0.5):
        for c in self.__dataframe.columns:
                miss = self.__dataframe[c].isnull().sum()
                missing_data_percent = round((miss/self.get_shape()[0])*100, 2)
                if missing_data_percent >= threshold*100:
                    self.drop_column(c)
    
    def get_index(self, as_list=True):
        if as_list is True:
            return self.__dataframe.index.to_list()
        return self.__dataframe.index
    
    def add_time_serie_row(self, date_column, value_column, value, date_format='%Y'):
        last_date = self.get_index()[-1] + timedelta(days=1)
        dataframe = DataFrame([{value_column: value, date_column: last_date}], data_type='dict')
        dataframe.to_time_series(date_column, value_column, one_row=True, date_format=date_format)
        self.append_dataframe(dataframe.get_dataframe())
        
    def set_generator(self, generator):
        self.__generator = generator

    def set_dataframe(self, data, data_type='df'):
        if data_type == 'matrix':
            index_name = [i for i in range(len(data))]
            colums_name = [i for i in range(len(data[0]))]
            self.__dataframe = pd.DataFrame(data=data, index=index_name, columns=colums_name)
        elif data_type == 'df':
            self.__dataframe = data

    def get_columns_types(self, show=True):
        types = self.get_dataframe().dtypes
        if show:
            print(types)
        return types
    
    def set_data_types(self, column_dict_types):
        self.__dataframe = self.get_dataframe().astype(column_dict_types)
        
    def set_same_type(self, same_type='float64'):
        """
        example of types: float64, object
        """
        for p in self.get_columns_names():
            self.set_column_type(p, same_type)

    def describe(self, show=True):
        description = self.get_dataframe().describe()
        if show:
            print(description)
        return description
    
    def reset_index(self, drop=True):
        if drop is True:
            self.set_dataframe(self.__dataframe.reset_index(drop=True))
        else:
            self.set_dataframe(self.__dataframe.reset_index())
            
    def get_dataframe_as_sparse_matrix(self):
        return scipy.sparse.csr_matrix(self.__dataframe.to_numpy())

    def get_column_as_list(self, column):
        return list(self.get_column(column))
    
    def get_column_as_joined_text(self, column):
        return ' '.join(list(self.get_column(column)))
    
    def rename_index(self, new_name):
        self.__dataframe.index.rename(new_name, inplace=True)
        return self.get_dataframe()

    def get_term_doc_matrix_as_df(self, text_column_name, vectorizer_type='count'):
        corpus = list(self.get_column(text_column_name))
        indice = ['doc' + str(i) for i in range(len(corpus))]
        v = Vectorizer(corpus, vectorizer_type=vectorizer_type)
        self.set_dataframe(DataFrame(v.get_sparse_matrix().toarray(), v.get_features_names(),
                                      line_index=indice, data_type='matrix').get_dataframe())

    def get_dataframe_from_dic_list(self, dict_list):
        v = DictVectorizer()
        matrice = v.fit_transform(dict_list)
        self.__vectorizer = v
        self.set_dataframe(DataFrame(matrice.toarray(), v.get_feature_names()).get_dataframe())

    def check_decision_function_on_column(self, column, decision_func):
        if all(self.get_column(column).apply(decision_func)):
            return True
        return False
    
    def show_word_occurrences_plot(self, column_name, most_common=50):
        """Generating word occurrences plot from a column

        Args:
            column_name (_type_): column to be used
            most_common (int, optional): number of most frequent term to use. Defaults to 50.
        """
        text = self.get_column_as_joined_text(column_name)
        counter = Counter(text.split(' '))
        data = DataFrame(counter.most_common(most_common), ['term', 'count'], data_type='dict', data_types_in_order=[str,int])
        chart = Chart(data.get_dataframe(), column4x='term', chart_type='bar')
        chart.add_data_to_show('count')
        chart.config('Term occurrences bar chart', 'Terms', 'Occurrences', titile_font_size=30,)
        chart.show()

    def set_dataframe_index(self, liste_indices):
        self.__dataframe.index = liste_indices

    def get_shape(self):
        return self.__dataframe.shape

    def set_column(self, column_name, new_column):
        self.__dataframe[column_name] = new_column

    def set_column_type(self, column, column_type):
        self.__dataframe[column] = self.__dataframe[column].astype(column_type)

    def get_lines_columns(self, lines, columns):
        if Lib.check_all_elements_type(columns, str):
            return self.get_dataframe().loc[lines, columns]
        return self.get_dataframe().iloc[lines, columns]
    
    def get_n_rows_as_dataframe(self, number_of_row=10):
        """
        give a negative value if you want begin from last row
        """
        if number_of_row < 0:
            return self.get_dataframe().tail(abs(number_of_row))
        else:
            return self.get_dataframe().head(number_of_row)

    def get_column(self, column):
        return self.get_dataframe()[column]
    
    def get_columns(self, columns_names_as_list):
        return self.get_dataframe()[columns_names_as_list]

    def rename_columns(self, column_dict_or_all_list, all_columns=False):
        if all_columns is True:
            types = {}
            self.__dataframe.columns = column_dict_or_all_list
            for p in column_dict_or_all_list:
                types[p] = str
            self.__dataframe = self.get_dataframe().astype(types)
        else:
            self.get_dataframe().rename(columns=column_dict_or_all_list, inplace=True) 

    def add_column(self, column, column_name):
        y = column
        if (not isinstance(column, pd.core.series.Series or not isinstance(column, pd.core.frame.DataFrame))):
            y = np.array(column)
            y = np.reshape(y, (y.shape[0],))
            y = pd.Series(y, self.get_index())
        self.__dataframe[column_name] = y
        
    def add_transformed_columns(self, dest_column_name="new_column", transformation_rule="okk*2"):
        columns_names = self.get_columns_names()
        columns_dict = {}
        for column_name in columns_names:
            if column_name in transformation_rule:
                columns_dict.update({column_name: self.get_column(column_name)})
        y_transformed = eval(transformation_rule, columns_dict)
        self.__dataframe[dest_column_name] = y_transformed
        
    def add_one_value_column(self, column_name, value, length=None):
        if length is not None:
            y = np.zeros(length)
            y.fill(value)
        else:
            y = np.zeros((self.get_shape()[0]))
            y.fill(value)
        self.__dataframe[column_name] = y
        return self.get_dataframe()
        
    def get_dataframe(self):
        return self.__dataframe

    def request(self, select, order_by=None, ascending=None):
        if order_by is not None:
            self.__dataframe = self.__dataframe.sort_values(order_by, ascending=ascending)
        return self.__dataframe[select]

    def contains(self, column, regex):
        return self.get_dataframe()[column].str.contains(regex)

    def to_upper_column(self, column):
        self.set_column(column, self.get_column(column).str.upper())

    def to_lower_column(self, column):
        self.set_column(column, self.get_column(column).str.lower())

    def sub(self, column, pattern, replacement):
        self.__dataframe = self.get_dataframe()[column].str.replace(pattern, replacement)

    def drop_column(self, column_name):
        """Drop a given column from the dataframe given its name

        Args:
            column (str): name of the column to drop

        Returns:
            [dataframe]: the dataframe with the column dropped
        """
        self.__dataframe = self.__dataframe.drop(column_name, axis=1)
        return self.__dataframe
        
    def drop_index(self, drop=False):
        if drop is True:
            self.__dataframe.reset_index(drop=True, inplace=True) 
        else:
            self.__dataframe.reset_index(drop=False, inplace=True) 
        
    def drop_columns(self, columns_names_as_list):
        for p in columns_names_as_list:
            self.__dataframe = self.__dataframe.drop(p, axis=1)
        return self.__dataframe
    
    def reorder_columns(self, new_order_as_list):
        self.__dataframe.reindex_axis(new_order_as_list, axis=1)
        return self.__dataframe
            
    def keep_columns(self, columns_names_as_list):
        for p in self.get_columns_names():
            if p not in columns_names_as_list:
                self.__dataframe = self.__dataframe.drop(p, axis=1)
        return self.__dataframe

    def add_row(self, row_as_dict):
        self.__dataframe = self.get_dataframe().append(row_as_dict, ignore_index=True)

    def pivot(self, index_columns_as_list, column_columns_as_list, column_of_values, agg_func):
        return self.get_dataframe().pivot_table(index=index_columns_as_list, columns=column_columns_as_list, values=column_of_values, aggfunc=agg_func)

    def group_by(self, column):
        self.set_dataframe(self.get_dataframe().groupby(column).count())
        
    def check_missed_data(self, column=None):
        if column is not None:
            if any(pd.isna(self.get_dataframe()[column])) is True:
                print("Missed data found in column " + column)
            else:
                print("No missed data in column " + column)
        else:
            for c in self.__dataframe.columns:
                miss = self.__dataframe[c].isnull().sum()
                if miss>0:
                    missing_data_percent = round((miss/self.get_shape()[0])*100, 2)
                    print("{} has {} missing value(s) which represents {}% of dataset size".format(c,miss, missing_data_percent))
                else:
                    print("{} has NO missing value!".format(c))

    def missing_data(self, filling_dict_colmn_val=None, drop_row_if_nan_in_column=None):
        if drop_row_if_nan_in_column is not None:
            # a = a[~(np.isnan(a).all(axis=1))] # removes rows containing all nan
            self.set_dataframe(self.__dataframe[self.__dataframe[drop_row_if_nan_in_column].notna()])
            #self.__dataframe = self.__dataframe[~(np.isnan(self.__dataframe).any(axis=1))] # removes rows containing at least one nan
        else:
            self.get_dataframe().fillna(filling_dict_colmn_val, inplace=True)
        
    def get_row(self, row_index):
        if isinstance(row_index, int):
            return self.get_dataframe().iloc[row_index]
        return self.get_dataframe().loc[row_index]
    
    def replace_column(self, column, pattern, replacement, regex=False, number_of_time=-1, case_sensetivity=False):
        self.set_column(column, self.get_column(column).str.replace(pattern, replacement, regex=regex, n=number_of_time,
                                                                    case=case_sensetivity))

    def replace_num_data(self, val, replacement):
        self.get_dataframe().replace(val, replacement, inplace=True)

    def map_function(self, func, **kwargs):
        self.__dataframe = self.get_dataframe().applymap(func, **kwargs)

    def apply_fun_to_column(self, column, func, in_place=True):
        if in_place is True:
            self.set_column(column, self.get_column(column).apply(func))
        else:
            return self.get_column(column).apply(func)
        
    def convert_column_type(self, column_name, new_type='float64'):
        """Convert the type of the column

        Args:
            column_name (str): Name of the column to convert
            Retruns (dataframe): New dataframe after conversion
        """
        self.set_column(column_name, self.get_column(column_name).astype(new_type))
        return self.get_columns_types()
    
    def convert_dataframe_type(self, new_type='float64'):
        for p in self.get_columns_names():
            self.convert_column_type(p, new_type)
        return self.get_columns_types()

    def concatinate(self, dataframe, ignore_index=False, join='outer'):
        """conacatenate horizontally two dataframe

        Args:
            dataframe (dataframe): the destination dataframe 
            ignore_index (bool, optional): If True, do not use the index values along the concatenation axis. Defaults to False.
        """
        # 
        self.__dataframe = pd.concat([self.get_dataframe(), dataframe], axis=1, ignore_index=ignore_index, join=join)
    
    def append_dataframe(self, dataframe):
        # append dataset contents data_sets must have the same columns names
        self.__dataframe = self.__dataframe.append(dataframe)

    def intersection(self, dataframe, column):
        self.__dataframe = pd.merge(self.__dataframe, dataframe, on=column, how='inner')

    def left_join(self, dataframe, column):
        self.__dataframe = pd.merge(self.__dataframe, dataframe, on=column, how='left')

    def right_join(self, dataframe, column):
        self.__dataframe = pd.merge(self.__dataframe, dataframe, on=column, how='right')

    def eliminate_outliers_neighbors(self, n_neighbors=20, contamination=.05):
        outliers = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        self.__dataframe['inlier'] = outliers.fit_predict(self.get_dataframe())
        self.__dataframe = self.get_dataframe().loc[self.get_dataframe().inlier == 1,
                                                      self.get_dataframe().columns.tolist()]

    def get_pca(self, new_dim):
        # pca.explained_variance_ratio_ gain d'info pour chaque vecteur
        pca_model = PCA(n_components=new_dim)
        return pca_model.fit_transform(self.get_dataframe())

    def get_centre_reduite(self):
        sc = StandardScaler()
        return sc.fit_transform(X=self.get_dataframe())
    
    def column_to_standard_scale(self, column):
        sc = StandardScaler()
        columns_names = self.get_columns_names()
        dataframe_copy = self
        dataframe = DataFrame(sc.fit_transform(X=self.get_dataframe()), columns_names_as_list=columns_names, data_type='matrix')
        self.reindex_dataframe()
        dataframe_copy.set_column(column, dataframe.get_column(column))
        self.set_dataframe(dataframe_copy.get_dataframe())
    
    def s__column_to_min_max_scale(self, column):
        self.set_column(column, minmax_scale(self.get_column(column)))
        
    def column_to_min_max_scale(self, column):
        self.__vectorizer = MinMaxScaler() 
        dataframe_copy = self.get_dataframe()
        self.keep_columns([column])
        self.__vectorizer.fit(self.get_dataframe())
        scaled_column = self.__vectorizer.transform(self.get_dataframe())
        self.set_dataframe(dataframe_copy)
        self.set_column(column, scaled_column)
        return scaled_column
    
    def get_min_max_scaled_columns(self, columns_names_as_list):
        self.__vectorizer = MinMaxScaler() 
        dataframe_copy = self.get_dataframe()
        self.keep_columns(columns_names_as_list)
        self.__vectorizer.fit(self.get_dataframe())
        scaled_column = self.__vectorizer.transform(self.get_dataframe())
        self.set_dataframe(dataframe_copy)
        return scaled_column
    
    def k_get_min_max_scaled_dataframe(self):
        self.__vectorizer = MinMaxScaler()
        self.__vectorizer.fit(self.get_dataframe())
        scaled_dataframe = DataFrame(self.__vectorizer.transform(self.get_dataframe()), 
                                     data_type='matrix',
                                     columns_names_as_list=self.get_columns_names())
        return scaled_dataframe.get_dataframe()
    
    def get_min_max_scaled_dataframe(self):
        self.__vectorizer = MinMaxScaler()
        self.__vectorizer.fit(self.get_dataframe())
        return self.__vectorizer.transform(self.get_dataframe())
        
    def dataframe_to_min_max_scale(self):
        self.__vectorizer = MinMaxScaler()
        self.set_dataframe(self.__vectorizer.fit_transform(X=self.get_dataframe()))
        
    def get_inverse_transform(self, scaled_list):
        scaled_list = np.reshape(scaled_list, (len(scaled_list), 1))
        return self.__vectorizer.inverse_transform(scaled_list)
        
    def get_last_window_for_time_serie_as_list(self, column, window_size=3):
        #print(np.reshape(self.get_column(column).iloc[-window_size:].to_numpy(), (window_size, 1)))
        #print(self.__vectorizer.transform([np.array(self.get_column(column).iloc[-window_size:])]))
        return self.__vectorizer.transform(np.reshape(self.get_column(column).iloc[-window_size:].to_numpy(), (window_size, 1)))

    def write_column_in_file(self, column, path='data/out.csv'):
        Lib.write_liste_in_file(path, self.get_column(column).apply(str))

    def check_duplicated_rows(self):
        return any(self.get_dataframe().duplicated())

    def check_duplicated_in_column(self, column):
        return any(self.get_column(column).duplicated())

    def write_check_duplicated_column_result_in_file(self, column, path='data/latin_comments.csv'):
        Lib.write_liste_in_file(path, self.get_column(column).duplicated().apply(str))

    def write_files_grouped_by_column(self, column_index, dossier):
        for p in self.get_dataframe().values:
            Lib.write_line_in_file(dossier + str(p[0]).lower() + '.csv', p[column_index])

    def filter_dataframe(self, column, func_de_decision, in_place=True, *args):
        if in_place is True:
            if len(args) == 2:
                self.set_dataframe(
                    self.get_dataframe().loc[self.get_column(column).apply(func_de_decision, args=(args[0], args[1]))])
            else:
                self.set_dataframe(
                    self.get_dataframe().loc[self.get_column(column).apply(func_de_decision)])
        else:
            if len(args) == 2:
                return self.get_dataframe().loc[self.get_column(column).apply(func_de_decision, args=(args[0], args[1]))]
            else:
                return self.get_dataframe().loc[self.get_column(column).apply(func_de_decision)]

    def transform_column(self, column_to_trsform, column_src, fun_de_trasformation, in_place=True, *args):
        """_summary_

        Args:
            column_to_trsform (_type_): column to transform
            column_src (_type_): Column to use as a source for the transformation
            fun_de_trasformation (_type_): The function of transformation, if it has multiple arguments pass them as args:
            example: data.transform_column(column, column, Lib.remove_stopwords, True, stopwords)
            in_place (bool, optional): If true the changes will affect the original dataframe. Defaults to True.

        Returns:
            _type_: _description_
        """
        if in_place is True:
            if (len(args) != 0):
                self.set_column(column_to_trsform, self.get_column(column_src).apply(fun_de_trasformation, args=(args[0],)))
            else:
                self.set_column(column_to_trsform, self.get_column(column_src).apply(fun_de_trasformation))
        else:
            if (len(args) != 0):
                return self.get_column(column_src).apply(fun_de_trasformation, args=(args[0],))
            else:
                return self.get_column(column_src).apply(fun_de_trasformation)
            
    def to_no_accent_column(self, column):
        self.trasform_column(column, column, Lib.no_accent)
        self.set_column(column, self.get_column(column))

    def write_dataframe_in_file(self, out_file='data/out.csv', delimiter=','):
        Lib.write_liste_csv(self.get_dataframe().values, out_file, delimiter)

    def sort(self, by_columns_list, ascending=False):
        self.set_dataframe(self.get_dataframe().sort_values(by=by_columns_list, ascending=ascending,
                                                              na_position='first'))

    def count_occurence_of_each_row(self, column):
        return self.get_dataframe().pivot_table(index=[column], aggfunc='size')
    
    def get_distinct_values_as_list(self, column):
        return list(self.get_dataframe().pivot_table(index=[column], aggfunc='size').index)
    
    def column_to_numerical_values(self, column):
        maping = list(self.get_dataframe().pivot_table(index=[column], aggfunc='size').index)
        self.transform_column(column, column, lambda o : maping.index(o))
        return maping
    
    def reverse_column_from_numerical_values(self, column, maping):
        self.trasform_column(column, column, lambda o : maping[int(o)])

    def count_occurence_of_row_as_count_column(self, column):
        column_name = 'count'
        self.set_column(column_name, self.get_column(column).value_counts())
        self.trasform_column(column_name, column, lambda x:self.get_column(column).value_counts().get(x))
    
    def get_count_number_of_all_words(self, column):
        self.apply_fun_to_column(column, lambda x: len(x.split(' ')))
        return self.get_column(column).sum()
    
    def get_count_occurrence_of_value(self, column, value, case_sensitive=True):
        
        if case_sensitive:
            self.apply_fun_to_column(column, lambda x: x.split(' ').count(value))
            return self.get_column(column).sum()
        else:
            self.apply_fun_to_column(column, lambda x: list(map(str.lower, x.split(' '))).count(value))
            return self.get_column(column).sum()

    def count_true_decision_function_rows(self, column, decision_function):
        self.filter_dataframe(column, decision_function)
        
    def show_wordcloud(self, column):
        wordcloud = WordCloud(
            background_color='white',
            max_words=100,
            max_font_size=30,
            scale=3,
            random_state=1)
   
        wordcloud = wordcloud.generate(self.get_column_as_joined_text(column))
        fig = plt.figure(1, figsize=(12, 12))
        plt.axis('off')
        plt.imshow(wordcloud)
        plt.show()

    def reindex_dataframe(self, index_as_column_name=None, index_as_liste=None):
        if index_as_liste is not None:
            new_index = new_index = index_as_liste
            self.get_dataframe().index = new_index
        if index_as_column_name is not None:
            self.__dataframe.set_index(index_as_column_name, inplace=True)
        if index_as_column_name is None and index_as_liste is None:
            new_index = pd.Series(np.arange(self.get_shape()[0]))
            self.get_dataframe().index = new_index
    
    def get_columns_names(self):
        header = list(self.get_dataframe().columns)
        return header 
    
    def export(self, destination_path='data/json_dataframe.csv', type='csv'):
        if type == 'json':
            destination_path='data/json_dataframe.json'
            self.get_dataframe().to_json(destination_path)
            return 0
        self.get_dataframe().to_csv(destination_path)
        
    def sample(self, n=10, frac=None):
        if frac is not None:
            return self.get_dataframe().sample(n=frac)
        return self.get_dataframe().sample(n=n)

    """
    filter lines == WHERE
    print(data.get_dataframe().loc[data.get_dataframe().Gender == 'H', 'Name']) nom des gen qui ont Gender  == 'H'
    data.set_column('Gender', data.get_column('Gender').apply(okkk)) 
    def okkk(o):
    if o == '0':
        return 'OKK'
    return 'NOK'
    print(data.get_dataframe().loc[data.get_dataframe().Gender == 'H', 'Name'])
    data.set_column('Gender', data.get_column('Gender').apply(okkk)) select en respectant la fun okkk
    if o == '0':
        return True
    return False

    filter = data["Age"]=="Twenty five"

    # printing only filtered columns 
    data.where(filter).dropna() 
    
    
    
    In [13]: df.iloc[0]  # first row in a DataFrame
    Out[13]: 
    A    1
    B    2
    Name: a, dtype: int64
    
    In [14]: df['A'].iloc[0]  # first item in a Series (Column)
    Out[14]: 1
    """

    def show(self, number_of_row=None):
        if number_of_row is None:
            print(self.get_dataframe())
        elif number_of_row < 0:
            return self.get_dataframe().tail(abs(number_of_row)) 
        else:
            return self.get_dataframe().head(number_of_row) 

    def get_sliced_dataframe(self, line_tuple, columns_tuple):
        return self.get_dataframe().loc[line_tuple[0]:line_tuple[1], columns_tuple[0]: columns_tuple[1]]

    def eliminate_outliers_quantile(self, column, min_quantile, max_quantile):
        min_q, max_q = self.get_column(column).quantile(min_quantile), self.get_column(column).quantile(max_quantile)
        self.filter_dataframe(column, self.outliers_decision_function, min_q, max_q)

    def scale_column(self, column):
        max_column = self.get_column(column).describe()['max']
        self.transform_column(column, column, self.scale_trasform_fun, max_column)

    def drop_duplicated_rows(self, column):
        self.set_dataframe(self.__dataframe.drop_duplicates(subset=column, keep='first'))
        
    def plot_column(self, column):
        self.get_column(column).plot()
        plt.show()
        
    def plot_dataframe(self):
        self.get_dataframe().plot()
        plt.show()
        
    def to_numpy(self):
        return self.get_dataframe().values
    
    def info(self):
        return self.get_dataframe().info()
    
    def drop_rows_by_year(self, year=2020, in_place=True):
        year = int(year)
        if in_place is True:
            self.set_dataframe(self.get_dataframe()[self.get_index(as_list=False).year != year]) 
        else:
            return self.get_dataframe()[self.get_index(as_list=False).year != year]
            
    def keep_rows_by_year(self, year=2020, in_place=True):
        year = int(year)
        if in_place is True:
            self.set_dataframe(self.get_dataframe()[self.get_index(as_list=False).year == year]) 
        else:
            return self.get_dataframe()[self.get_index(as_list=False).year == year]
        
    def train_test_split(self, train_percent=0.8):
        seuil = ceil(self.get_shape()[0]*train_percent)
        train = self.get_dataframe().iloc[:seuil]
        test = self.get_dataframe().iloc[seuil:]
        return train, test
    
    def train_test_split_column(self, column, train_percent=0.8):
        seuil = ceil(self.get_shape()[0]*train_percent)
        train = self.get_column(column).iloc[:seuil]
        test = self.get_column(column).iloc[seuil:]
        return train, test
        
    def column_to_date(self, column, format='%Y-%m-%d %H:%M'):
        self.set_column(column, pd.to_datetime(self.get_column(column)))
        self.set_column(column, self.get_column(column).dt.strftime(format))
        self.set_column(column, pd.to_datetime(self.get_column(column)))
        
    def reformat_date_time(self, date_time_column_name, new_format='%Y-%m-%d %H:%M'):
        self.set_column(date_time_column_name, self.get_column(date_time_column_name).dt.strftime(new_format))
        return self.get_dataframe()
        
    def resample_timeseries(self, frequency='d', agg='mean'):
        if agg == 'sum':
            self.set_dataframe(self.__dataframe.resample(frequency).sum())
        if agg == 'mean':
            self.set_dataframe(self.__dataframe.resample(frequency).mean())
        return self.get_dataframe()
        
    def to_time_series(self, date_column, value_column, date_format='%Y-%m-%d', window_size=2, one_row=False):
        # when working with train test generators
        """def to_time_series_generators(self, date_column, time_series_column, date_format='%Y-%m-%d', window_size=2, train_percent=0.8):
        self.column_to_date(date_column, format=date_format)
        self.reindex_dataframe(self.get_column(date_column))
        self.drop_column(date_column)
        #dataframa.asfreq('d') # h hourly w weekly d normal daily b business day  m monthly a annualy
        train, test = self.train_test_split_column(time_series_column, 0.8)
        self.set_train_generator(TimeseriesGenerator(np.reshape(train.values,
                                                          (len(train),1)),
                                               np.reshape(train.values,
                                                          (len(train),1)),
                                               length=window_size,
                                               batch_size=1,
                                               ))
        self.set_test_generator(TimeseriesGenerator(np.reshape(test.values,
                                                          (len(test),1)),
                                               np.reshape(test.values,
                                                          (len(test),1)),
                                               length=window_size,
                                               batch_size=1,
                                               ))
        return self.get_train_generator(), self.get_test_generator()"""
        self.column_to_date(date_column, format=date_format)
        self.reindex_dataframe(self.get_column(date_column))
        self.keep_columns(value_column)
        if one_row is False:
            #dataframa.asfreq('d') # h hourly w weekly d normal daily b business day  m monthly a annualy
            self.set_generator(SG(self.get_min_max_scaled_dataframe(),
                                                self.get_min_max_scaled_dataframe(),
                                                length=window_size,
                                                batch_size=1,))
            """self.set_generator(
                TimeseriesGenerator(self.get_min_max_scaled_dataframe(), 
                                    self.get_min_max_scaled_dataframe(), 
                                    length=window_size, 
                                    length_output=7, 
                                    batch_size=1)"""
            return self.get_generator()
    
    def drop_rows(self, nbr_rows=1):
        """Drop the first nbr_rows of rows from the dataframe

        Args:
            nbr_rows (int, optional): if negative value is given then thelen the last nbr_rows. Defaults to 1.

        Returns:
            None
        """
        
        if nbr_rows < 0:
            self.set_dataframe(self.get_dataframe().iloc[:self.get_shape()[0]+nbr_rows])
        else:
            self.set_dataframe(self.get_dataframe().iloc[nbr_rows:])
            
    def drop_rows_by_indices(self, indexes_as_list=[0]):
        """Drop rows given their indexes

        Args:
            indexes_as_list (list, optional): [description]. Defaults to [0].
        """
        self.set_dataframe(self.get_dataframe().drop(indexes_as_list))
        
    def save(self, path=None):
        """Save a dataframe in pkl format for future use

        Args:
            path ([type], optional): link and name of storage file. If set to None, it will be dataframe.pkl.
        """
        if path is None:
            self.get_dataframe().to_pickle("dataframe.pkl")
        else:
            self.get_dataframe().to_pickle(path)
            
        
    def dataframe_skip_columns(self, intitial_index, final_index, step=2):
        self.set_dataframe(self.get_dataframe().loc[intitial_index:final_index:step])
        
    def shuffle_dataframe(self):
        self.set_dataframe(self.get_dataframe().sample(frac=1).reset_index(drop=True))
        
    def scale_columns(self, columns_names_as_list, scaler_type='min_max', in_place=True):
        """A method  to standardize the independent features present in the concerned columns in a fixed range.

        Args:
            column_name ([type]): 
            scaler_type (str, optional): ['min_max', 'standard', 'adjusted_log']. Defaults to 'min_max'.
            in_place (bool, optional): if False the modification do not  affects the original columns. Defaults to True.
        """
        if scaler_type == 'min_max':
            self.__vectorizer = MinMaxScaler() 
            dest_columns = self.get_columns(columns_names_as_list)
            dest_dataframe = DataFrame(self.__vectorizer.fit_transform(X=dest_columns), 
                                       line_index=self.get_index(),
                                       columns_names_as_list=columns_names_as_list, 
                                       data_type='matrix')
            self.drop_columns(columns_names_as_list)
            self.concatinate(dest_dataframe.get_dataframe())
            return dest_dataframe.get_dataframe()
        elif scaler_type == 'standard':
            self.__vectorizer = StandardScaler()
            dest_columns = self.get_columns(columns_names_as_list)
            dest_dataframe = DataFrame(self.__vectorizer.fit_transform(X=dest_columns), 
                                       line_index=self.get_index(),
                                       columns_names_as_list=columns_names_as_list, 
                                       data_type='matrix')
            self.drop_columns(columns_names_as_list)
            self.concatinate(dest_dataframe.get_dataframe())
            return dest_dataframe.get_dataframe()
        elif scaler_type == 'adjusted_log':
            def log_function(o, min_column):
                return np.log(1 + o - min_column)
            for name in columns_names_as_list:
                min_column = self.get_column(name).min()
                self.transform_column(name, name, log_function, min_column)
            return self.get_columns(columns_names_as_list)
                        
    def scale_dataframe(self, scaler_type='min_max', in_place=True):
        """A method  to standardize the independent features present in the dataframe in a fixed range.

        Args:
            column_name ([type]): 
            scaler_type (str, optional): ['min_max', 'standard', 'adjusted_log']. Defaults to 'min_max'.
            in_place (bool, optional): if False the modification do not  affects the dataframe. Defaults to True.
        """
        if scaler_type == 'min_max':
            self.__vectorizer = MinMaxScaler() 
            column_names = self.get_columns_names()
            self.set_dataframe(DataFrame(self.__vectorizer.fit_transform(X=self.get_dataframe()), 
                                       line_index=self.get_index(),
                                       columns_names_as_list=column_names, 
                                       data_type='matrix').get_dataframe())
            return self.get_dataframe()
        elif scaler_type == 'standard':
            self.__vectorizer = StandardScaler() 
            self.set_dataframe(self.__vectorizer.fit_transform(X=self.get_dataframe()))
            return self.get_dataframe()
        elif scaler_type == 'adjusted_log':
            def log_function(o, min_column):
                return np.log(1 + o - min_column)
            for name in self.get_columns_names():
                min_column = self.get_column(name).min()
                self.transform_column(name, name, log_function, min_column)
            return self.get_dataframe()
        
    def load_dataset(self, dataset='iris'):
        """
        boston: Load and return the boston house-prices dataset (regression)
        iris: Load and return the iris dataset (classification).
        """
        if dataset == 'boston':
            data = load_boston()
            x = data.data
            y = data.target
            features_names = data.feature_names
            self.set_dataframe(x, data_type='matrix')
            self.rename_columns(features_names, all_columns=True)
            self.add_column(y,'house_price')
            
        elif dataset == 'iris':
            data = load_iris(as_frame=True)
            x = data.data
            y = data.target
            self.set_dataframe(x)
            self.add_column(y,'target')            
        
    @staticmethod
    def outliers_decision_function(o, min_quantile, max_quantile):
        if min_quantile < o < max_quantile:
            return True
        return False
    
    @staticmethod
    def generate_datetime_range(starting_datetime='2013-01-01', periods=365, freq='1H'):
        return pd.date_range(starting_datetime, periods=periods, freq=freq)

    @staticmethod
    def scale_trasform_fun(o, max_column):
        return o / max_column