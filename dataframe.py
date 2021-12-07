from datetime import timedelta
from math import ceil
import pandas as pd
import scipy.sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from lib import *
from vectorizer import Vectorizer
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.compose import ColumnTransformer
from keras.preprocessing.sequence import TimeseriesGenerator as SG
from sklearn.datasets import load_iris, load_boston
import numpy as np

class DataFrame:
    """DataFrame class for dealing with exploratry data analysis and data preprocessing
    """
    __vectorizer = None
    __generator = None

    def __init__(self, data_link=None, columns_names_as_list=None, data_types_in_order=None, delimiter=',',
                 file_type='csv', line_index=None, skip_empty_line=False, sheet_name='Sheet1'):
        if data_link is not None:
            if file_type == 'csv':
                self.__dataframe = pd.read_csv(data_link, encoding='utf-8', delimiter=delimiter, low_memory=False, error_bad_lines=False, skip_blank_lines=False)
            elif file_type == 'json':
                self.__dataframe = pd.read_json(data_link, encoding='utf-8')
            elif file_type == 'xls':
                self.__dataframe = pd.read_excel(data_link, sheet_name=sheet_name)
            elif file_type == 'dict':
                self.__dataframe = pd.DataFrame.from_dict(data_link)
            elif file_type == 'matrix':
                index_name = [i for i in range(len(data_link))]
                colums_name = [i for i in range(len(data_link[0]))]
                self.__dataframe = pd.DataFrame(data=data_link, index=index_name, columns=colums_name)
                """data = array([['','Col1','Col2'],['Row1',1,2],['Row2',3,4]])
                pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:]) """
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
        """

        Returns:
            (generator): return the data generator build using the dataframe
        """
        return self.__generator
    
    def get_index(self):
        return self.__dataframe.index.to_list()
    
    def add_time_serie_row(self, date_column, value_column, value, date_format='%Y'):
        """[summary]

        Args:
            date_column (str): The column containing the date
            value_column (str): The column containing values of the time series
            value ([type]): new value to add
            date_format (str, optional): The RegEx patern of date format. Defaults to '%Y'.
        """
        last_date = self.get_index()[-1] + timedelta(days=1)
        dataframe = DataFrame([{value_column: value, date_column: last_date}], file_type='dict')
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

    def get_data_types(self, show=True):
        """[summary]

        Args:
            show (bool, optional): if True printing the dataframe in console. Defaults to True.

        Returns:
            list: a list of data types of all columns in the dataframe
        """
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
        """[summary]

        Args:
            show (bool, optional): if True printing the dataframe in console. Defaults to True.

        Returns:
            [type]: [description]
        """
        description = self.get_dataframe().describe()
        if show:
            print(description)
        return description
    
    def reset_index(self):
        self.set_dataframe(self.__dataframe.reset_index())

    def get_dataframe_as_sparse_matrix(self):
        return scipy.sparse.csr_matrix(self.__dataframe.to_numpy())

    def get_column_as_list(self, column):
        return list(self.get_column(column))
    
    def get_column_as_joined_text(self, column):
        return ' '.join(list(self.get_column(column)))

    def get_term_doc_matrix_as_df(self, vectorizer_type='count'):
        """Build a term matrix document from a text column

        Args:
            vectorizer_type (str, optional): weighing scheme to use. Defaults to 'count'.
        """
        corpus = list(self.get_column('comment'))
        indice = ['doc' + str(i) for i in range(len(corpus))]
        v = Vectorizer(corpus, vectorizer_type=vectorizer_type)
        self.set_dataframe(DataFrame(v.get_sparse_matrix().toarray(), v.get_features_names(),
                                      line_index=indice, file_type='ndarray').get_dataframe())

    def get_dataframe_from_dic_list(self, dict_list):
        v = DictVectorizer()
        matrice = v.fit_transform(dict_list)
        self.__vectorizer = v
        self.set_dataframe(DataFrame(matrice.toarray(), v.get_feature_names()).get_dataframe())

    def check_decision_function_on_column(self, column, decision_func):
        if all(self.get_column(column).apply(decision_func)):
            return True
        return False

    def set_dataframe_index(self, liste_indices):
        self.__dataframe.index = liste_indices

    def get_shape(self):
        return self.__dataframe.shape

    def set_column(self, column_name, new_column):
        self.__dataframe[column_name] = new_column

    def set_column_type(self, column, column_type):
        self.__dataframe[column] = self.__dataframe[column].astype(column_type)

    def get_lines_columns(self, lines, columns):
        if check_all_elements_type(columns, str):
            return self.get_dataframe().loc[lines, columns]
        return self.get_dataframe().iloc[lines, columns]
    
    def get_n_rows_as_dataframe(self, number_of_row=10):
        """return n rows as new dataframe: give a negative value if you want begin from last row

        Args:
            number_of_row (int, optional): Defaults to 10.

        Returns:
            (dataframe): the new dataframe
        """
        if number_of_row < 0:
            return self.get_dataframe().tail(abs(number_of_row))
        else:
            return self.get_dataframe().head(number_of_row)

    def get_column(self, column):
        return self.get_dataframe()[column]

    def rename_columns(self, column_dict_or_all_list, all_columns=False):
        """[summary]

        Args:
            column_dict_or_all_list (dictionary or list): dictionary of new column names or a list of all columns names
            all_columns (bool, optional): [description]. Defaults to False.
        """
        if all_columns is True:
            types = {}
            self.__dataframe.columns = column_dict_or_all_list
            for p in column_dict_or_all_list:
                types[p] = str
            self.__dataframe = self.get_dataframe().astype(types)
        else:
            self.get_dataframe().rename(columns=column_dict_or_all_list, inplace=True)

    def add_column(self, column, column_name):
        """add a new column to the dataframe

        Args:
            column (Series, ndarray or a list): column to be added
            column_name (str): name of new added column
        """
        y = column
        if (not isinstance(column, pd.core.series.Series or not isinstance(column, pd.core.frame.DataFrame))):
            y = np.array(column)
            y = np.reshape(y, (y.shape[0],))
            y = pd.Series(y)
        self.__dataframe[column_name] = y

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

    def drop_column(self, column):
        self.__dataframe = self.__dataframe.drop(column, axis=1)
        
    def drop_index(self):
        self.__dataframe.reset_index(drop=True, inplace=True) 
        
    def drop_columns(self, columns_names_as_list):
        for p in columns_names_as_list:
            self.__dataframe = self.__dataframe.drop(p, axis=1)
            
    def keep_columns(self, columns_names_as_list):
        for p in self.get_columns_names():
            if p not in columns_names_as_list:
                self.__dataframe = self.__dataframe.drop(p, axis=1)

    def add_row(self, row_as_dict):
        self.__dataframe = self.get_dataframe().append(row_as_dict, ignore_index=True)

    def pivot(self, index_columns_as_list, column_columns_as_list, column_of_values, agg_func):
        return self.get_dataframe().pivot_table(index=index_columns_as_list, columns=column_columns_as_list, values=column_of_values, aggfunc=agg_func)

    def group_by(self, column):
        self.set_dataframe(self.get_dataframe().groupby(column).count())
        
    def check_missed_data(self, column=None):
        """Retrning statistics about missing data

        Args:
            column (str, optional): column name to check if None take all column in concidiration. Defaults to None.
        """
        if column is not None:
            if any(pd.isna(self.get_dataframe()[column])) is True:
                print("Missed data found in column " + column)
            else:
                print("No missed data in column " + column)
        else:
            print(self.get_dataframe().isna().any())
            for c in self.__dataframe.columns:
                miss = self.__dataframe[c].isnull().sum()
                if miss>0:
                    missing_data_percent = round((miss/self.get_shape()[0])*100, 2)
                    print("{} has {} missing value(s) which represents {}% of dataset size".format(c,miss, missing_data_percent))
                else:
                    print("{} has NO missing value!".format(c))

    def missing_data(self, filling_dict_colmn_val=None, drop_row_if_nan_in_column=None):
        """Filling missing data

        Args:
            filling_dict_colmn_val (dict, optional): dictionary of missing values mapping. Defaults to None.
            drop_row_if_nan_in_column (str, optional): delete the entire row if NaN is found in this column. Defaults to None.
        """
        if drop_row_if_nan_in_column is not None:
            # a = a[~(np.isnan(a).all(axis=1))] # removes rows containing all nan
            self.set_dataframe(self.__dataframe[self.__dataframe[drop_row_if_nan_in_column].notna()])
            #self.__dataframe = self.__dataframe[~(np.isnan(self.__dataframe).any(axis=1))] # removes rows containing at least one nan
        else:
            self.get_dataframe().fillna(filling_dict_colmn_val, inplace=True)
        
    def get_row(self, row_index):
        return self.get_dataframe().iloc[row_index]

    def replace_column(self, column, pattern, replacement, regex=False, number_of_time=-1, case_sensetivity=False):
        self.set_column(column, self.get_column(column).str.replace(pattern, replacement, regex=regex, n=number_of_time,
                                                                    case=case_sensetivity))

    def replace_num_data(self, val, replacement):
        self.get_dataframe().replace(val, replacement, inplace=True)

    def map_function(self, func):
        self.__dataframe = self.get_dataframe().applymap(func)

    def apply_fun_to_column(self, column, func, in_place=True):
        if in_place is True:
            self.set_column(column, self.get_column(column).apply(func))
        else:
            return self.get_column(column).apply(func)
        
    def convert_column_type(self, column, new_type):
        self.set_column(column, self.get_column(column).astype(new_type))

    def union(self, dataframe):
        # conacatener deux dataframe avec column pas unique
        self.__dataframe = pd.concat([self.get_dataframe(), dataframe], axis=1)

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
        """Delete outliers from the dataframe based on number of neighbors

        Args:
            n_neighbors (int, optional): . Defaults to 20.
            contamination (float, optional): . Defaults to .05.
        """
        outliers = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        self.__dataframe['inlier'] = outliers.fit_predict(self.get_dataframe())
        self.__dataframe = self.get_dataframe().loc[self.get_dataframe().inlier == 1,
                                                      self.get_dataframe().columns.tolist()]

    def get_pca(self, new_dim):
        """Apply PCA method to the dataframe

        Args:
            new_dim (int):  dimensions of new dataframe i.e number of features

        Returns:
            [type]: [description]
        """
        # pca.explained_variance_ratio_ gain d'info pour chaque vecteur
        pca_model = PCA(n_components=new_dim)
        return pca_model.fit_transform(self.get_dataframe())

    def get_centre_reduite(self):
        """Standar scaler

        Returns:
            dataframe: scaled dataframe
        """
        sc = StandardScaler()
        return sc.fit_transform(X=self.get_dataframe())
    
    def column_to_standard_scale(self, column):
        sc = StandardScaler()
        columns_names = self.get_columns_names()
        dataframe_copy = self
        dataframe = DataFrame(sc.fit_transform(X=self.get_dataframe()), columns_names_as_list=columns_names, file_type='matrix')
        self.reindex_dataframe()
        dataframe_copy.set_column(column, dataframe.get_column(column))
        self.set_dataframe(dataframe_copy.get_dataframe())
    
    def s__column_to_min_max_scale(self, column):
        self.set_column(column, minmax_scale(self.get_column(column)))
        
    def column_to_min_max_scale(self, column_name):
        """Apply min/max scale to a column

        Args:
            column_name (str): name of the column to scale

        Returns:
            series: new scaled column
        """
        self.__vectorizer = MinMaxScaler() 
        dataframe_copy = self.get_dataframe()
        self.keep_columns([column_name])
        self.__vectorizer.fit(self.get_dataframe())
        scaled_column = self.__vectorizer.transform(self.get_dataframe())
        self.set_dataframe(dataframe_copy)
        self.set_column(column_name, scaled_column)
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
                                     file_type='matrix',
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
        write_liste_in_file(path, self.get_column(column).apply(str))

    def check_duplicated_rows(self):
        return any(self.get_dataframe().duplicated())

    def check_duplicated_in_column(self, column):
        return any(self.get_column(column).duplicated())

    def write_check_duplicated_column_result_in_file(self, column, path='data/latin_comments.csv'):
        write_liste_in_file(path, self.get_column(column).duplicated().apply(str))

    def write_files_grouped_by_column(self, column_index, dossier):
        for p in self.get_dataframe().values:
            write_line_in_file(dossier + str(p[0]).lower() + '.csv', p[column_index])

    def filter_dataframe(self, column_name, decision_function, in_place=True, *args):
        """[summary]

        Args:
            column_name (str): name of the column to filter
            decision_function (function): function that take in parameter a row and check a condition upon it
            in_place (bool, optional): if True apply the decision function to inner dataframe. Defaults to True.

        Returns:
            dataframe: filtred dataframe
        """
        if in_place is True:
            if len(args) == 2:
                self.set_dataframe(
                    self.get_dataframe().loc[self.get_column(column_name).apply(decision_function, args=(args[0], args[1]))])
            else:
                self.set_dataframe(
                    self.get_dataframe().loc[self.get_column(column_name).apply(decision_function)])
        else:
            if len(args) == 2:
                return self.get_dataframe().loc[self.get_column(column_name).apply(decision_function, args=(args[0], args[1]))]
            else:
                return self.get_dataframe().loc[self.get_column(column_name).apply(decision_function)]

    def transform_column(self, column_to_trsform, column_src, fun_de_trasformation, *args):
        if (len(args) != 0):
            self.set_column(column_to_trsform, self.get_column(column_src).apply(fun_de_trasformation, args=(args[0],)))
        else:
            self.set_column(column_to_trsform, self.get_column(column_src).apply(fun_de_trasformation))
            
    def to_no_accent_column(self, column):
        self.trasform_column(column, column, no_accent)
        self.set_column(column, self.get_column(column))

    def write_dataframe_in_file(self, out_file='data/out.csv', delimiter=','):
        write_liste_csv(self.get_dataframe().values, out_file, delimiter)

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

    def count_occurence_of_row_as_count_column(self, column_name):
        """Calculate the occurrence of each row in the dataframe as a new column

        Args:
            column_name (str): name of destination column
        """
        new_column_name = 'count'
        print(self.get_column(column_name).value_counts())
        self.set_column(new_column_name, self.get_column(column_name).value_counts())
        self.trasform_column(new_column_name, column_name, lambda x:self.get_column(column_name).value_counts().get(x))
    
    def get_count_number_of_all_words(self, column_name):
        self.apply_fun_to_column(column_name, lambda x: len(x.split(' ')))
        return self.get_column(column_name).sum()
    
    def get_count_occurrence_of_a_value(self, column_name, value, case_sensitive=True):
        """Count the occurrence of a value in a given column

        Args:
            column_name (str): name of destination column
            value (any): The value to be counted
            case_sensitive (bool, optional): in case of textual data take in concidiration the case sensitive. Defaults to True.

        Returns:
            int: the occurrence number
        """
        if case_sensitive:
            self.apply_fun_to_column(column_name, lambda x: x.split(' ').count(value))
            return self.get_column(column_name).sum()
        else:
            self.apply_fun_to_column(column_name, lambda x: list(map(str.lower, x.split(' '))).count(value))
            return self.get_column(column_name).sum()

    def count_true_decision_function_rows(self, column, decision_function):
        self.filter_dataframe(column, decision_function)
        print(self.get_shape()[0])
        
    def show_wordcloud(self, column_name):
        """Build the WordCloud of a textual column

        Args:
            column_name (str): name of destination column
        """
        wordcloud = WordCloud(
            background_color='white',
            max_words=100,
            max_font_size=30,
            scale=3,
            random_state=1)
        wordcloud = wordcloud.generate(self.get_column_as_joined_text(column_name))
        fig = plt.figure(1, figsize=(12, 12))
        plt.axis('off')
        plt.imshow(wordcloud)
        plt.show()

    def reindex_dataframe(self, index_as_liste=None, index_as_column_name=None):
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
        self.trasform_column(column, column, self.scale_trasform_fun, max_column)

    def drop_duplicated_rows(self, column):
        self.set_dataframe(self.__dataframe.drop_duplicates(subset=column, keep='first'))
        
    def plot_column(self, column):
        self.get_column(column).plot()
        
    def plot_dataframe(self):
        self.get_dataframe().plot()
        
    def to_numpy(self):
        return self.get_dataframe().values
    
    def info(self):
        return self.get_dataframe().info()
    
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
        
    def resample_timeseries(self, frequency='d', agg='mean'):
        if agg == 'sum':
            self.set_dataframe(self.__dataframe.resample(frequency).sum())
        if agg == 'mean':
            self.set_dataframe(self.__dataframe.resample(frequency).mean())
        return 0
        
    def to_time_series(self, date_column_name, value_column_name, date_format='%Y-%m-%d', window_size=2, one_row=False):
        """Convert a dataframe to a time series

        Args:
            date_column_name (str): The column containing the date
            value_column (str): The column containing values
            date_format (str, optional): date pattern. Defaults to '%Y-%m-%d'.
            window_size (int, optional): length of the window. Defaults to 2.
            one_row (bool, optional): [description]. Defaults to False.

        Returns:
            data generator: generated data
        """
        self.column_to_date(date_column_name, format=date_format)
        self.reindex_dataframe(self.get_column(date_column_name))
        self.keep_columns(value_column_name)
        if one_row is False:
            #dataframa.asfreq('d') # h hourly w weekly d normal daily b business day  m monthly a annualy
            self.set_generator(SG(self.get_min_max_scaled_dataframe(),
                                                self.get_min_max_scaled_dataframe(),
                                                length=window_size,
                                                batch_size=1,))
            
            return self.get_generator()
    
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