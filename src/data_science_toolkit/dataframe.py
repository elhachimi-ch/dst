from datetime import timedelta
from math import ceil
import pandas as pd
from pyparsing import col
import scipy.sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .lib import Lib
from .vectorizer import Vectorizer
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.compose import ColumnTransformer
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator as SG
from sklearn.datasets import load_iris
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
                 data_type='csv', has_header=True, line_index=None, skip_empty_line=False, sheet_name=0,
                 skip_rows=None, **kwargs
                 ):
        
        if data_link is not None:
            if data_type == 'csv':
                if has_header is True:
                    self.dataframe = pd.read_csv(data_link, encoding='utf-8', delimiter=delimiter, 
                                               low_memory=False, on_bad_lines='skip', skip_blank_lines=False,
                                               skiprows=skip_rows, **kwargs)
                else:
                    self.dataframe = pd.read_csv(data_link, encoding='utf-8', delimiter=delimiter, 
                                               low_memory=False, on_bad_lines='skip', skip_blank_lines=False,
                                               header=None)
            elif data_type == 'json':
                self.dataframe = pd.read_json(data_link, encoding='utf-8')
            elif data_type == 'xls':
                self.dataframe = pd.read_excel(data_link, sheet_name=sheet_name,
                                                 skiprows=skip_rows)
            elif data_type == 'pkl':
                self.dataframe = pd.read_pickle(data_link)
            elif data_type == 'dict':
                self.dataframe = pd.DataFrame.from_dict(data_link)
            elif data_type == 'matrix':
                index_name = [i for i in range(len(data_link))]
                colums_name = [i for i in range(len(data_link[0]))]
                self.dataframe = pd.DataFrame(data=data_link, index=index_name, columns=colums_name)
            elif data_type == 'list':
                y = data_link
                if (not isinstance(y, pd.core.series.Series or not isinstance(y, pd.core.frame.DataFrame))):
                    y = np.array(y)
                    y = np.reshape(y, (y.shape[0],))
                    y = pd.Series(y)
                self.dataframe = pd.DataFrame()
                if columns_names_as_list is not None:
                    self.dataframe[columns_names_as_list[0]] = y
                else:
                    self.dataframe['0'] = y
                    
                
                """data = array([['','Col1','Col2'],['Row1',1,2],['Row2',3,4]])
                pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:]) """
            elif data_type == 'df':
                self.dataframe = data_link
            types = {}
            if data_types_in_order is not None and columns_names_as_list is not None:
                self.dataframe.columns = columns_names_as_list
                for i in range(len(columns_names_as_list)):
                    types[columns_names_as_list[i]] = data_types_in_order[i]
            elif columns_names_as_list is not None:
                self.dataframe.columns = columns_names_as_list
                for p in columns_names_as_list:
                    types[p] = str

            self.dataframe = self.get_dataframe().astype(types)

            if line_index is not None:
                self.dataframe.index = line_index
        else:
            self.dataframe = pd.DataFrame()
        
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
        return self.dataframe
    
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
        self.add_column(column_name, series)
        return self.dataframe
    
    def drop_full_nan_columns(self):
        for c in self.dataframe.columns:
                miss = self.dataframe[c].isnull().sum()
                missing_data_percent = round((miss/self.get_shape()[0])*100, 2)
                if missing_data_percent == 100:
                    self.drop_column(c)
                    
    def drop_columns_with_nan_threshold(self, threshold=0.5):
        for c in self.dataframe.columns:
                miss = self.dataframe[c].isnull().sum()
                missing_data_percent = round((miss/self.get_shape()[0])*100, 2)
                if missing_data_percent >= threshold*100:
                    self.drop_column(c)
    
    def get_index(self, as_list=True):
        if as_list is True:
            return self.dataframe.index.to_list()
        return self.dataframe.index
    
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
            self.dataframe = pd.DataFrame(data=data, index=index_name, columns=colums_name)
        elif data_type == 'df':
            self.dataframe = data

    def get_columns_types(self, show=True):
        types = self.get_dataframe().dtypes
        if show:
            print(types)
        return types
    
    def set_data_types(self, column_dict_types):
        self.dataframe = self.get_dataframe().astype(column_dict_types)
        
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
            self.set_dataframe(self.dataframe.reset_index(drop=True))
        else:
            self.set_dataframe(self.dataframe.reset_index())
            
    def get_dataframe_as_sparse_matrix(self):
        return scipy.sparse.csr_matrix(self.dataframe.to_numpy())

    def get_column_as_list(self, column):
        return list(self.get_column(column))
    
    def get_column_as_joined_text(self, column):
        return ' '.join(list(self.get_column(column)))
    
    def rename_index(self, new_name):
        self.dataframe.index.rename(new_name, inplace=True)
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
        self.dataframe.index = liste_indices

    def get_shape(self):
        return self.dataframe.shape

    def set_column(self, column_name, new_column):
        self.dataframe[column_name] = new_column

    def set_column_type(self, column, column_type):
        self.dataframe[column] = self.dataframe[column].astype(column_type)

    def get_lines_columns(self, lines, columns):
        if Lib.check_all_elements_type(columns, str):
            return self.get_dataframe().loc[lines, columns]
        return self.get_dataframe().iloc[lines, columns]
    
    def get_rows(self, 
                 nbr_of_rows=None, 
                 start_index=None, 
                 end_index=None, 
                 index_type=None, 
                 frequency='d', 
                 datetime_format='%Y-%m-%d %H:%M:%S'):
        """
        give a negative value if you want begin from last row
        """
        
        if index_type == 'int':
            if start_index is not None and end_index is not None:
                return self.get_dataframe().iloc[start_index:end_index]
            elif start_index is not None and nbr_of_rows is not None:
                return self.get_dataframe().iloc[start_index:start_index+nbr_of_rows]
            elif start_index is None and end_index is None and nbr_of_rows is not None:
                if nbr_of_rows < 0:
                    return self.get_dataframe().tail(abs(nbr_of_rows))
                else:
                    return self.get_dataframe().head(nbr_of_rows)
        elif index_type == 'datetime':
            if start_index is not None and end_index is not None:
                return self.get_dataframe().loc[start_index:end_index]
            elif start_index is not None and nbr_of_rows is not None:
                start_index = Lib.to_datetime(start_index, datetime_format)
                end_index = start_index + datetime.timedelta(nbr_of_rows)
                return self.get_dataframe().loc[start_index:end_index]
            elif start_index is None and end_index is None and nbr_of_rows is not None:
                if nbr_of_rows < 0:
                    return self.get_dataframe().tail(abs(nbr_of_rows))
                else:
                    return self.get_dataframe().head(nbr_of_rows)
        else:
            if start_index is not None and end_index is not None:
                return self.get_dataframe().loc[start_index:end_index]
            elif start_index is not None and nbr_of_rows is not None:
                return self.get_dataframe().loc[start_index:start_index+nbr_of_rows]
            elif start_index is None and end_index is None and nbr_of_rows is not None:
                if nbr_of_rows < 0:
                    return self.get_dataframe().tail(abs(nbr_of_rows))
                else:
                    return self.get_dataframe().head(nbr_of_rows)
            
        

    def get_column(self, column):
        return self.get_dataframe()[column]
    
    def get_columns(self, columns_names_as_list):
        return self.get_dataframe()[columns_names_as_list]
    
    def add_noise(self, column_name, num_noises=100):
        """
        Adds random noise to a Pandas time series.
        
        Parameters:
        ts (pandas.Series): the time series to which to add noise
        num_noises (int): the number of noise values to add to the time series
        
        Returns:
        pandas.Series: the time series with noise added
        """
        # Calculate the range of the time series data
        data_range = self.get_column(column_name).max() - self.get_column(column_name).min()
        
        # Add the specified number of random noise values
        for i in range(num_noises):
            # Generate a random index within the time series
            rand_index = self.dataframe.sample().index[0]
            
            # Generate a random noise value within the data range
            noise_value = np.random.uniform(low=-data_range, high=data_range)
            
            # Add the noise value to the time series at the random date
            self.set_row(column_name, rand_index, noise_value)
        
        return 0

    def rename_columns(self, column_dict_or_all_list, all_columns=False):
        if all_columns is True:
            types = {}
            self.dataframe.columns = column_dict_or_all_list
            for p in column_dict_or_all_list:
                types[p] = str
            self.dataframe = self.get_dataframe().astype(types)
        else:
            self.get_dataframe().rename(columns=column_dict_or_all_list, inplace=True) 

    def add_column(self, column_name, column):
        y = column
        if (not isinstance(column, pd.core.series.Series or not isinstance(column, pd.core.frame.DataFrame))):
            y = np.array(column)
            y = np.reshape(y, (y.shape[0],))
            if len(self.get_index()) == 0:
                y = pd.Series(y)
            else:
                y = pd.Series(y, self.get_index())
        self.dataframe[column_name] = y
        
    def add_transformed_columns(self, dest_column_name="new_column", transformation_rule="okk*2"):
        columns_names = self.get_columns_names()
        columns_dict = {}
        operations = {'sqrt': sqrt, 
         'pow': power,
         'exp': exp,
         }
        columns_dict.update(operations)
        for column_name in columns_names:
            if column_name in transformation_rule:
                columns_dict.update({column_name: self.get_column(column_name)})
        y_transformed = eval(transformation_rule, columns_dict)
        self.dataframe[dest_column_name] = y_transformed
        
    def add_one_value_column(self, column_name, value, length=None):
        if length is not None:
            y = np.zeros(length)
            y.fill(value)
        else:
            y = np.zeros((self.get_shape()[0]))
            y.fill(value)
        self.dataframe[column_name] = y
        return self.get_dataframe()
        
    def get_dataframe(self):
        return self.dataframe

    def request(self, select, order_by=None, ascending=None):
        if order_by is not None:
            self.dataframe = self.dataframe.sort_values(order_by, ascending=ascending)
        return self.dataframe[select]

    def contains(self, column, regex):
        return self.get_dataframe()[column].str.contains(regex)

    def to_upper_column(self, column):
        self.set_column(column, self.get_column(column).str.upper())
        
    def combine_date_and_time_columns(self, 
                                      date_column_name='date', 
                                      time_column_name='time', 
                                      new_date_time_column_name='date_time',
                                      date_time_format='%Y-%m-%d %H:%M:%S'):
        self.add_column(new_date_time_column_name, self.get_column(date_column_name).astype(str) + ' ' + self.get_column(time_column_name).astype(str))
        self.column_to_date(new_date_time_column_name, format=date_time_format)
        
    def to_lower_column(self, column):
        self.set_column(column, self.get_column(column).str.lower())

    def sub(self, column, pattern, replacement):
        self.dataframe = self.get_dataframe()[column].str.replace(pattern, replacement)

    def drop_column(self, column_name):
        """Drop a given column from the dataframe given its name

        Args:
            column (str): name of the column to drop

        Returns:
            [dataframe]: the dataframe with the column dropped
        """
        self.dataframe = self.dataframe.drop(column_name, axis=1)
        return self.dataframe
        
    def index_to_column(self, column_name=None):
        self.dataframe.reset_index(drop=False, inplace=True) 
        if column_name is not None:
            self.rename_columns({'index': column_name})
        
    def drop_columns(self, columns_names_as_list):
        for p in columns_names_as_list:
            self.dataframe = self.dataframe.drop(p, axis=1)
        return self.dataframe
    
    def reorder_columns(self, new_order_as_list):
        self.dataframe.reindex_axis(new_order_as_list, axis=1)
        return self.dataframe
            
    def keep_columns(self, columns_names_as_list):
        for p in self.get_columns_names():
            if p not in columns_names_as_list:
                self.dataframe = self.dataframe.drop(p, axis=1)
        return self.dataframe

    def add_row(self, row_as_dict, index=None):
        if index is not None:
            row = pd.DataFrame(row_as_dict, index=[index])
            self.dataframe = pd.concat([self.dataframe.iloc[:index], row, self.dataframe.iloc[index:]]).reset_index(drop=True)
            #self.reset_index()
        else:
            self.dataframe = self.get_dataframe().append(row_as_dict, ignore_index=True)

    def pivot(self, index_columns_as_list, column_columns_as_list, column_of_values, agg_func):
        return self.get_dataframe().pivot_table(index=index_columns_as_list, columns=column_columns_as_list, values=column_of_values, aggfunc=agg_func)

    def group_by(self, column_name):
        self.set_dataframe(self.get_dataframe().groupby(column_name).count())
        
    def get_nan_indexes_of_column(self, column_name):
        return list(self.get_dataframe().loc[pd.isna(self.get_column(column_name)), :].index)
        
    def missing_data_checking(self, column_name=None):
        if column_name is not None:
            if any(pd.isna(self.get_dataframe()[column_name])) is True:
                miss = self.dataframe[column_name].isnull().sum()
                missing_data_percent = round((miss/self.get_shape()[0])*100, 2)
                print("{} has {} missing value(s) which represents {}% of dataset size".format(column_name, miss, missing_data_percent))
            else:
                print("No missed data in column " + column_name)
        else:
            miss = []
            for c in self.dataframe.columns:
                miss_by_column = self.dataframe[c].isnull().sum()
                if miss_by_column>0:
                    missing_data_percent = round((miss_by_column/self.get_shape()[0])*100, 2)
                    print("{} has {} missing value(s) which represents {}% of dataset size".format(c, miss_by_column, missing_data_percent))
                else:
                    print("{} has NO missing value!".format(c))
                miss.append(miss_by_column)
        return miss
    
    def missing_data_column_percent(self, column_name):
        return self.dataframe[column_name].isnull().sum()/self.get_shape()[0]
    
    def get_missing_data_indexes_in_column(self, column_name):
        return self.dataframe[self.dataframe[column_name].isnull()].index.tolist()

    def missing_data(self, drop_row_if_nan_in_column=None, filling_dict_colmn_val=None, method='ffill',
                     column_to_fill='Ta', date_column_name=None):
        if filling_dict_colmn_val is None and drop_row_if_nan_in_column is None:
            if method == 'ffill':
                self.get_dataframe().fillna(method='pad', inplace=True)
            elif method == 'bfill':
                self.get_dataframe().fillna(method='backfill', inplace=True)
            elif method == 'era5_land':
                if column_to_fill == 'Ta':
                    era5_land_variables = ['2m_temperature']
                elif column_to_fill == 'Hr':
                    era5_land_variables = ['2m_temperature', '2m_dewpoint_temperature']
                elif column_to_fill == 'Rg':
                    era5_land_variables = ['surface_solar_radiation_downwards']
                
                from gis import GIS
                import cdsapi
                c = cdsapi.Client()

                if date_column_name is not None:
                   self.reindex_dataframe(date_column_name)

                indexes = []
                for p in self.get_missing_data_indexes_in_column(column_to_fill):
                    if isinstance(p, str) is True:
                        indexes.append(datetime.datetime.strptime(p, '%Y-%m-%d %H:%M:%S'))
                    else:
                        indexes.append(p)
                    
                    
                years = set()
                for p in indexes:
                    years.add(p.year)     
                    
                years = list(years)
                print("Found missing data for {} in year(s): {}".format(column_to_fill, years))  
                for y in years:
                    missing_data_dict = {}
                    missing_data_dict['month'] = set()   
                    missing_data_dict['day'] = set() 
                    for p in indexes:
                        if p.year == y:
                            missing_data_dict['month'].add(p.strftime('%m'))
                            missing_data_dict['day'].add(p.strftime('%d'))
                    missing_data_dict['month'] = list(missing_data_dict['month'])
                    missing_data_dict['day'] = list(missing_data_dict['day'])

                    if os.path.exists("E:\projects\pythonsnippets\era5_r3_" + column_to_fill + '_' + str(y) + ".grib") is False:
                        c.retrieve(
                            'reanalysis-era5-land',
                            {
                                'format': 'grib',
                                'variable': era5_land_variables,
                                'year': str(y),
                                'month':  missing_data_dict['month'],
                                'day': missing_data_dict['day'],
                                'time': [
                                    '00:00', '01:00', '02:00',
                                    '03:00', '04:00', '05:00',
                                    '06:00', '07:00', '08:00',
                                    '09:00', '10:00', '11:00',
                                    '12:00', '13:00', '14:00',
                                    '15:00', '16:00', '17:00',
                                    '18:00', '19:00', '20:00',
                                    '21:00', '22:00', '23:00',
                                ],
                                'area': [
                                    31.66749781, -7.593311291, 31.66749781,
                                    -7.593311291,
                                ],
                            },
                            'era5_r3_' + column_to_fill + '_' + str(y) +'.grib')
                
                gis = GIS()

                data = DataFrame(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill + '_' + str(years[0]) + ".grib", "ta"),
                                data_type="df")
                data.reset_index()
                data.resample_timeseries(skip_rows=2)
                data.reindex_dataframe("valid_time")
                if column_to_fill == 'Ta':
                    data.keep_columns(['t2m'])
                    for y in years[1:]:
                        data_temp = DataFrame(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill + '_' + str(y) + ".grib", "ta"),
                                            data_type='df')
                        data_temp.reset_index()
                        data_temp.resample_timeseries(skip_rows=2)
                        data_temp.reindex_dataframe("valid_time")
                        data_temp.keep_columns(['t2m'])
                        data.append_dataframe(data_temp.get_dataframe())
                    
                    data.transform_column('t2m', 't2m', lambda o: o - 273.15)
                    nan_indices = self.get_nan_indexes_of_column(column_to_fill)
                    for p in nan_indices:
                        self.set_row('Ta', p, data.get_row(p)['t2m'])
                    print('Imputation of missing data for Ta from ERA5-Land was done!')
                    
                elif column_to_fill == 'Hr':
                    data.keep_columns(['t2m', 'd2m'])
                    for y in years[1:]:
                        data_temp = DataFrame(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill + '_' + str(y) + ".grib", "ta"),
                                            data_type='df')
                        data_temp.reset_index()
                        data_temp.resample_timeseries(skip_rows=2)
                        data_temp.reindex_dataframe("valid_time")
                        data_temp.keep_columns(['t2m', 'd2m'])
                        data.append_dataframe(data_temp.get_dataframe())
                    data.transform_column('t2m', 't2m', lambda o: o - 273.15)
                    data.transform_column('d2m', 'd2m', lambda o: o - 273.15)
                    data.add_transformed_columns('era5_hr', '100*exp(-((243.12*17.62*t2m)-(d2m*17.62*t2m)-d2m*17.62*(243.12+t2m))/((243.12+t2m)*(243.12+d2m)))')
                    nan_indices = self.get_missing_data_indexes_in_column(column_to_fill)
                    for p in nan_indices:
                        self.set_row('Hr', p, data.get_row(p)['era5_hr'])
                    
                    print('Imputation of missing data for Hr from ERA5-Land was done!')
                elif column_to_fill == 'Rg':
                    data.keep_columns(['ssrd'])
                    for y in years[1:]:
                        data_temp = DataFrame(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill + '_' + str(y) + ".grib", "ta"),
                                            data_type='df')
                        data_temp.reset_index()
                        data_temp.resample_timeseries(skip_rows=2)
                        data_temp.reindex_dataframe("valid_time")
                        data_temp.keep_columns(['ssrd'])
                        data.append_dataframe(data_temp.get_dataframe())
                    
                    
                    l = []
                    for p in data.get_index():
                        if p.hour == 1:
                            new_value = data.get_row(p)['ssrd']/3600
                        else:
                            try:
                                previous_hour = data.get_row(p-timedelta(hours=1))['ssrd']
                            except KeyError: # if age is not convertable to int
                                previous_hour = data.get_row(p)['ssrd']
                                
                            new_value = (data.get_row(p)['ssrd'] - previous_hour)/3600
                        l.append(new_value)

                    data.add_column('rg', l)
                    data.keep_columns(['rg'])
                    data.rename_columns({'rg': 'ssrd'})
                    data.transform_column('ssrd', 'ssrd', lambda o : o if abs(o) < 1500 else 0 )    
                    data.export('rg.csv', index=True)
                    nan_indices = self.get_nan_indexes_of_column(column_to_fill)
                    for p in nan_indices:
                        self.set_row('Rg', p, data.get_row(p)['ssrd'])
                    
                    print('Imputation of missing data for Rg from ERA5-Land was done!')
        
        if filling_dict_colmn_val is not None:
            self.get_dataframe().fillna(filling_dict_colmn_val, inplace=True)
            
        if drop_row_if_nan_in_column is not None:
            if drop_row_if_nan_in_column == 'all':
                for p in self.get_columns_names():
                    self.set_dataframe(self.dataframe[self.dataframe[p].notna()])
            else:
                # a = a[~(np.isnan(a).all(axis=1))] # removes rows containing all nan
                self.set_dataframe(self.dataframe[self.dataframe[drop_row_if_nan_in_column].notna()])
                #self.dataframe = self.dataframe[~(np.isnan(self.dataframe).any(axis=1))] # removes rows containing at least one nan
          
       
            
        
    def get_row(self, row_index):
        if isinstance(row_index, int):
            return self.get_dataframe().iloc[row_index]
        return self.get_dataframe().loc[row_index]
    
    def set_row(self, column_name, row_index, new_value):
        if isinstance(row_index, int):
            self.dataframe[column_name].iloc[row_index] = new_value
        self.dataframe[column_name].loc[row_index] = new_value
    
    def replace_column(self, column, pattern, replacement, regex=False, number_of_time=-1, case_sensetivity=False):
        self.set_column(column, self.get_column(column).str.replace(pattern, replacement, regex=regex, n=number_of_time,
                                                                    case=case_sensetivity))

    def replace_num_data(self, val, replacement):
        self.get_dataframe().replace(val, replacement, inplace=True)

    def map_function(self, func, **kwargs):
        self.dataframe = self.get_dataframe().applymap(func, **kwargs)

    def apply_fun_to_column(self, column, func, in_place=True,):
        if in_place is True:
            self.set_column(column, self.get_column(column).apply(func))
        else:
            return self.get_column(column).apply(func)
        
    def add_column_based_on_function(self, column_name, func_accepting_row):
        self.add_column(column_name, self.get_dataframe().apply(func_accepting_row, axis=1))
        
    def convert_column_type(self, column_name, new_type='float64'):
        """Convert the type of the column

        Args:
            column_name (str): Name of the column to convert
            Retruns (dataframe): New dataframe after conversion
        """
        self.set_column(column_name, self.get_column(column_name).astype(new_type))
    
    def convert_dataframe_type(self, new_type='float64', ):
        for p in self.get_columns_names():
            self.convert_column_type(p, new_type)

    def concatinate(self, dataframe, ignore_index=False, join='outer'):
        """conacatenate horizontally two dataframe

        Args:
            dataframe (dataframe): the destination dataframe 
            ignore_index (bool, optional): If True, do not use the index values along the concatenation axis. Defaults to False.
        """
        # 
        self.dataframe = pd.concat([self.get_dataframe(), dataframe], axis=1, ignore_index=ignore_index, join=join)
    
    def append_dataframe(self, dataframe):
        # append dataset contents data_sets must have the same columns names
        self.dataframe = pd.concat([self.dataframe, dataframe], ignore_index=True)
        

    def join(self, dataframe, on_column='index', how='inner'):
        if on_column == 'index':
           self.dataframe = pd.merge(self.get_dataframe(), dataframe, left_index=True, right_index=True, how=how)
        else:
            self.dataframe = pd.merge(self.dataframe, dataframe, on=on_column, how=how)

    def left_join(self, dataframe, column):
        self.dataframe = pd.merge(self.dataframe, dataframe, on=column, how='left')

    def right_join(self, dataframe, column):
        self.dataframe = pd.merge(self.dataframe, dataframe, on=column, how='right')

    def eliminate_outliers_neighbors(self, n_neighbors=20, contamination=.05):
        outliers = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        self.dataframe['inlier'] = outliers.fit_predict(self.get_dataframe())
        self.dataframe = self.get_dataframe().loc[self.get_dataframe().inlier == 1,
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

    def filter_dataframe(self, column, decision_function, in_place=True, *args):
        if in_place is True:
            if len(args) == 2:
                self.set_dataframe(
                    self.get_dataframe().loc[self.get_column(column).apply(decision_function, args=(args[0], args[1]))])
            else:
                self.set_dataframe(
                    self.get_dataframe().loc[self.get_column(column).apply(decision_function)])
        else:
            if len(args) == 2:
                return self.get_dataframe().loc[self.get_column(column).apply(decision_function, args=(args[0], args[1]))]
            else:
                return self.get_dataframe().loc[self.get_column(column).apply(decision_function)]
            
    def select_datetime_range(self, start_datetime, end_datetime, in_place=True, *args):
        if in_place is True:
            self.dataframe = self.dataframe.loc[start_datetime:end_datetime]
        else:
            return self.dataframe.loc[start_datetime:end_datetime]

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
        self.transform_column(column_name, column, lambda x:self.get_column(column).value_counts().get(x))
        return self.get_dataframe()
    
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
        
    def add_artificial_missing_data(self, column_name, nbr_missing_data=31, method='continious'):
        """
        Fill a randomly selected period with NaN values in a specified column of a pandas dataframe.

        Parameters:
            df (pandas.DataFrame): The input dataframe.
            column (str): The name of the column to fill with NaN values.
            period (str): The length of the period to fill with NaN values, in pandas frequency string format (e.g., 'D' for day, 'W' for week, 'M' for month).

        Returns:
            None
        """
        
        filled_indices = []
        if method == 'continious':
            random_index = self.dataframe.sample().index[0]
            #self.dataframe[random_index:random_index+nbr_missing_data][column_name] = np.nan 
            previous_data = self.dataframe.loc[(self.dataframe.index >= random_index) & (self.dataframe.index < random_index + nbr_missing_data), column_name]
            self.dataframe.loc[(self.dataframe.index >= random_index) & (self.dataframe.index < random_index + nbr_missing_data), column_name] = np.nan
            # Fill the selected period with NaN values in the specified column
            #df.loc[(df.index >= random_period) & (df.index < random_period + pd.Timedelta(period)), column] = np.nan
            filled_indices = range(random_index, random_index + nbr_missing_data) 
        else:
            for  i in range(nbr_missing_data):
                random_index = self.dataframe.sample().index[0]
                self.dataframe.loc[self.dataframe.index == random_index, column_name] = np.nan
                filled_indices.append(random_index)
        return previous_data
            
    def plot_column(self, column_name, x_column_name='index', 
                    x_label='Date & time',
                    y_label=None,
                    x_label_rotation=0,
                    y_label_rotation=0,
                    save_fig=False,
                    savefig_path='out.png',
                    date_format_x_axis=None
                    ):
        """_summary_

        Args:
            column_name (_type_): _description_
            x_column_name (str, optional): _description_. Defaults to 'index'.
            x_label (str, optional): _description_. Defaults to 'Date & time'.
            y_label (_type_, optional): _description_. Defaults to None.
            x_label_rotation (int, optional): _description_. Defaults to 0.
            y_label_rotation (int, optional): _description_. Defaults to 0.
            save_fig (bool, optional): _description_. Defaults to False.
            savefig_path (str, optional): _description_. Defaults to 'out.png'.
            date_format_x_axis (str, optional): _description_. Example to '%m-%d'.
        """
        fig, ax = plt.subplots()
        
        if date_format_x_axis is not None:
            import matplotlib.dates as mdate
            # format the x-axis tick labels
            date_format = mdate.DateFormatter(date_format_x_axis)
            ax.xaxis.set_major_formatter(date_format)
            
        if x_column_name == 'index':
            ax.plot(self.get_index(), self.get_column(column_name))
        else:
            ax.plot(self.get_column(x_column_name), self.get_column(column_name))
        # set the axis labels and title
        ax.set_xlabel(x_label)
        if y_label is None:
            y_label = column_name
        ax.set_ylabel(y_label)
        ax.tick_params(axis='x', rotation=x_label_rotation)
        ax.tick_params(axis='y', rotation=y_label_rotation)
        plt.tight_layout()
        if save_fig is True:
            import matplotlib as mpl
            mpl.rcParams['agg.path.chunksize'] = 10000
            fig.savefig(savefig_path, dpi=720)
        plt.show()
        
    def split_export(self, percentage=0.8, train_out_file="train.csv", test_out_file="test.csv"):
        train = self.dataframe.iloc[:int(percentage*self.get_shape()[0]), :]
        test = self.dataframe.iloc[int(percentage*self.get_shape()[0]):, :]
        train.to_csv(train_out_file, index=False)
        test.to_csv(test_out_file, index=False) 
        
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
            self.dataframe.set_index(index_as_column_name, inplace=True)
        if index_as_column_name is None and index_as_liste is None:
            new_index = pd.Series(np.arange(self.get_shape()[0]))
            self.get_dataframe().index = new_index
    
    def get_columns_names(self):
        header = list(self.get_dataframe().columns)
        return header 
    
    def export_column(self, column_name, out_file='out.csv'):
        self.get_column(column_name).to_csv(out_file, index=False)
    
    def export(self, destination_path='data/json_dataframe.csv', type='csv', index=False):
        if type == 'json':
            destination_path='data/json_dataframe.json'
            self.get_dataframe().to_json(destination_path)
        elif type == 'csv':
            self.get_dataframe().to_csv(destination_path, index=index)
        elif type == 'pkl':
            self.get_dataframe().to_pickle(destination_path)
        print('DataFrame exported successfully to /' + destination_path)
        
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
            return self.get_dataframe()
        elif number_of_row < 0:
            return self.get_dataframe().tail(abs(number_of_row)) 
        else:
            return self.get_dataframe().head(number_of_row) 

    def get_sliced_dataframe(self, line_tuple, columns_tuple):
        return self.get_dataframe().loc[line_tuple[0]:line_tuple[1], columns_tuple[0]: columns_tuple[1]]
    
    def compare_two_times(self, time_series1_column_name, time_series2_column_name):
        from sklearn.metrics import r2_score, mean_squared_error 
        import math
        comparaison_dict = {}
        comparaison_dict['RMSE'] = math.sqrt(mean_squared_error(self.get_column(time_series1_column_name), self.get_column(time_series2_column_name)))
        comparaison_dict['R²'] = r2_score(self.get_column(time_series1_column_name), self.get_column(time_series2_column_name))
        return comparaison_dict

    def eliminate_outliers_quantile(self, column, min_quantile, max_quantile):
        min_q, max_q = self.get_column(column).quantile(min_quantile), self.get_column(column).quantile(max_quantile)
        self.filter_dataframe(column, self.outliers_decision_function, min_q, max_q)

    def scale_column(self, column):
        max_column = self.get_column(column).describe()['max']
        self.transform_column(column, column, self.scale_trasform_fun, max_column)

    def drop_duplicated_rows(self, column):
        self.set_dataframe(self.dataframe.drop_duplicates(subset=column, keep='first'))
        
    def drop_duplicated_indexes(self):
        self.dataframe = self.dataframe[~self.dataframe.index.duplicated(keep='first')]
        
    def plot_dataframe(self):
        self.get_dataframe().plot()
        plt.show()
        
    def to_numpy(self):
        return self.get_dataframe().values
    
    @staticmethod
    def generate_datetime_range_dataframe(starting_datetime='2013-01-01 00:00:00', 
                                          end_datetime='2013-12-31 00:00:00', 
                                          periods=None, 
                                          freq='1H'):
        dataframe = DataFrame()
        dataframe.set_dataframe_index(DataFrame.generate_datetime_range(
            starting_datetime=starting_datetime, 
            end_datetime=end_datetime, 
            periods=periods,
            freq=freq))
        return dataframe.get_dataframe()
    
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
    
    @staticmethod
    def get_elevation_and_latitude(lat, lon):
        """
        Returns the elevation (in meters) and latitude (in degrees) for a given set of coordinates.
        Uses the Open Elevation API (https://open-elevation.com/) to obtain the elevation information.
        """
        # 'https://api.open-elevation.com/api/v1/lookup?locations=10,10|20,20|41.161758,-8.583933'
        url = f'https://api.opentopodata.org/v1/aster30m?locations={lat},{lon}'
        response = requests.get(url)
        print(response.json())
        data = response.json()
        elevation = data['results'][0]['elevation']
        #latitude = data['results'][0]['latitude']
        return elevation
    
    @staticmethod
    def et0_penman_monteith(row):
        # input variables
        # T = 25.0  # air temperature in degrees Celsius
        # RH = 60.0  # relative humidity in percent
        # u2 = 2.0  # wind speed at 2 m height in m/s
        # Rs = 15.0  # incoming solar radiation in MJ/m2/day
        # lat = 35.0  # latitude in degrees
        
        ta_max, ta_min, rh_max, rh_min, u2_mean, rs_mean, lat, elevation, doy =  row['ta_max'], row['ta_min'], row['rh_max'], row['rh_min'], row['u2_mean'], row['rg_mean'], row['lat'], row['elevation'], row['doy']
        
        # constants
        ALBEDO = 0.23  # Albedo coefficient for grass reference surface
        GSC = 0.082  # solar constant in MJ/m2/min
        SIGMA = 0.000000004903  # Stefan-Boltzmann constant in MJ/K4/m2/day
        G = 0  # Soil heat flux density (MJ/m2/day)
        z = 2 # Convert wind speed measured at different heights above the soil surface to wind speed at 2 m above the surface, assuming a short grass surface.

        # convert units
        rs_mean *= 0.0864  # convert watts per square meter to megajoules per square meter 0.0288 = 60x60x8hours or 0.0864 for 24 hours
        ta_mean = (ta_max + ta_min) / 2
        ta_max_kelvin = ta_max + 273.16  # air temperature in Kelvin
        ta_min_kelvin = ta_min + 273.16  # air temperature in Kelvin
        
        # saturation vapor pressure in kPa
        es_max = 0.6108 * math.exp((17.27 * ta_max) / (ta_max + 237.3))
        es_min = 0.6108 * math.exp((17.27 * ta_min) / (ta_min + 237.3))
        es = (es_max + es_min) / 2
        
        # actual vapor pressure in kPa
        ea_max_term = es_max * (rh_min / 100)
        ea_min_term = es_min * (rh_max / 100)
        ea = (ea_max_term + ea_min_term) / 2
        
        # in the absence of rh_max and rh_min
        #ea = (rh_mean / 100) * es
        
        # when using equipement where errors in estimation rh min can be large or when rh data integrity are in doubt use only rh_max term
        #ea = ea_min_term  
        
        delta = (4098 * (0.6108 * math.exp((17.27 * ta_mean) / (ta_mean + 237.3)))) / math.pow((ta_mean + 237.3), 2) # slope of the vapor pressure curve in kPa/K
        
        
        atm_pressure = math.pow(((293.0 - (0.0065 * elevation)) / 293.0), 5.26) * 101.3
        # psychrometric constant in kPa/K
        gamma = 0.000665 * atm_pressure
        
        # Calculate u2
        u2 = u2_mean * (4.87 / math.log((67.8 * z) - 5.42))
        
        # Calculate extraterrestrial radiation
        dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
        d = 0.409*math.sin( (((2*math.pi) / 365) * doy) - 1.39)
        
        
        # sunset hour angle
        phi = math.radians(lat)
        omega = math.acos(-math.tan(phi)*math.tan(d))
        ra = ((24 * 60)/math.pi) * GSC * dr * ((omega * math.sin(phi) * math.sin(d)) + (math.cos(phi) * math.cos(d) * math.sin(omega)))
        
        # Calculate clear sky solar radiation
        rso = (0.75 + (2e-5 * elevation)) * ra
        
        # Calculate net solar shortwave radiation 
        rns = (1 - ALBEDO) * rs_mean
        
        # Calculate net longwave radiation
        rnl = SIGMA * (((math.pow(ta_max_kelvin, 4) + math.pow(ta_min_kelvin, 4)) / 2) * (0.34 - (0.14 * math.sqrt(ea))) * ((1.35 * (rs_mean / rso)) - 0.35))
        
        # Calculate net radiation
        rn = rns - rnl
        
        
        # decompose et0 to two terms to facilitate the calculation
        """rng = 0.408 * rn
        radiation_term = ((delta) / (delta + (gamma * (1 + 0.34 * u2)))) * rng
        pt = (gamma) / (delta + gamma * (1 + (0.34 * u2)))
        tt = ((900) / (ta_mean + 273) ) * u2
        wind_term = pt * tt * (es - ea)
        et0 = radiation_term + wind_term """
        
        et0 = ((0.408 * delta * (rn - G)) + gamma * ((900 / (ta_mean + 273)) * u2 * (es - ea))) / (delta + (gamma * (1 + 0.34 * u2)))

        # output result
        return et0

    @staticmethod
    def et0_hargreaves(row):
        ta_mean, ta_max, ta_min, lat, doy =  row['ta_mean'], row['ta_max'], row['ta_min'], row['lat'], row['doy']
        
        # constants
        GSC = 0.082  # solar constant in MJ/m2/min

        # Calculate extraterrestrial radiation
        dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
        d = 0.409*math.sin( (((2*math.pi) / 365) * doy) - 1.39)

        # sunset hour angle
        phi = math.radians(lat)
        omega = math.acos(-math.tan(phi)*math.tan(d))
        ra = ((24 * 60)/math.pi) * GSC * dr * ((omega * math.sin(phi) * math.sin(d)) + (math.cos(phi) * math.cos(d) * math.sin(omega)))
        
        et0 = 0.0023 * (ta_mean + 17.8) * (ta_max - ta_min) ** 0.5 * 0.408 * ra

        return et0
    
    def et0_estimation(self, 
                       air_temperture_column_name='ta',
                       global_solar_radiation_column_name='rs',
                       air_relative_humidity_column_name='rh',
                       wind_speed_column_name='ws',
                       date_time_column_name='date_time',
                       latitude=31.65410805,
                       longitude=-7.603140831,
                       method='pm',
                       in_place=True
                       ):
        
        et0_data = DataFrame()
        et0_data.add_column('ta_mean', self.resample_timeseries(in_place=False)[air_temperture_column_name])
        et0_data.add_column('ta_max', self.resample_timeseries(in_place=False, agg='max')[air_temperture_column_name])
        et0_data.add_column('ta_min', self.resample_timeseries(in_place=False, agg='min')[air_temperture_column_name], )
        et0_data.add_column('rh_max', self.resample_timeseries(in_place=False, agg='max')[air_relative_humidity_column_name])
        et0_data.add_column('rh_min', self.resample_timeseries(in_place=False, agg='min')[air_relative_humidity_column_name])
        et0_data.add_column('rh_mean', self.resample_timeseries(in_place=False)[air_relative_humidity_column_name])
        et0_data.add_column('u2_mean', self.resample_timeseries(in_place=False)[wind_speed_column_name])
        et0_data.add_column('rg_mean', self.resample_timeseries(in_place=False)[global_solar_radiation_column_name])
        et0_data.index_to_column()
        et0_data.add_doy_column('date_time')
        et0_data.add_one_value_column('elevation', DataFrame.get_elevation_and_latitude(latitude, longitude))
        et0_data.add_one_value_column('lat', latitude)
        
        if method == 'pm':
            et0_data.add_column_based_on_function('et0_pm', DataFrame.et0_penman_monteith)
        elif method == 'hargreaves':
            et0_data.add_column_based_on_function('et0_hargreaves', DataFrame.et0_hargreaves)
            
        if in_place == True:
            self.dataframe = et0_data.get_dataframe()
            
        return et0_data.get_dataframe()
        
        
    def column_to_date(self, column_name, format='%Y-%m-%d %H:%M:%S', extraction_func=None):
        if extraction_func is None:
            self.set_column(column_name, pd.to_datetime(self.get_column(column_name)))
            self.set_column(column_name, self.get_column(column_name).dt.strftime(format))
            self.set_column(column_name, pd.to_datetime(self.get_column(column_name)))
        else:
            self.transform_column(column_name, column_name, extraction_func)
        
    
        
    def datetime_reformate(self, date_time_column_name, new_format='%Y-%m-%d %H:%M:%S'):
        self.set_column(date_time_column_name, self.get_column(date_time_column_name).dt.strftime(new_format))
        return self.get_dataframe()

    def resample_timeseries(self, 
                            frequency='d', 
                            agg='mean', 
                            skip_rows=None, 
                            intitial_index=0, 
                            between_time_tuple=None,
                            date_column_name=None,
                            in_place=True):
        if in_place is True:
            if skip_rows is not None:
                self.set_dataframe(self.get_dataframe().loc[intitial_index:self.get_shape()[0]:skip_rows])
                self.reset_index()
            else:
                if date_column_name is not None:
                    self.reindex_dataframe(date_column_name)
                    
                if between_time_tuple is not None:
                    temp_time_series = temp_time_series.between_time(between_time_tuple[0], between_time_tuple[1])
                    
                if agg == 'sum':
                    self.set_dataframe(self.dataframe.resample(frequency).sum())
                if agg == 'mean':
                    self.set_dataframe(self.dataframe.resample(frequency).mean())
                if agg == 'max':
                    self.set_dataframe(self.dataframe.resample(frequency).max())
                if agg == 'min':
                    self.set_dataframe(self.dataframe.resample(frequency).min())
                if agg == 'median':
                    self.set_dataframe(self.dataframe.resample(frequency).median())
                if agg == 'std':
                    self.set_dataframe(self.dataframe.resample(frequency).std())
                if agg == 'var':
                    self.set_dataframe(self.dataframe.resample(frequency).var())
                if agg == 'ffill':
                    self.set_dataframe(self.dataframe.resample(frequency).ffill())
                if agg == 'bfill':
                    self.set_dataframe(self.dataframe.resample(frequency).bfill())
                else:
                    self.set_dataframe(self.dataframe.resample(frequency).mean())
            return self.get_dataframe()
        else:
            if skip_rows is not None:
                self.set_dataframe(self.get_dataframe().loc[intitial_index:self.get_shape()[0]:skip_rows])
            else:
                if date_column_name is not None:
                    self.reindex_dataframe(date_column_name)
                    
                temp_time_series = self.dataframe
                
                if between_time_tuple is not None:
                    temp_time_series = temp_time_series.between_time(between_time_tuple[0], between_time_tuple[1])

                if agg == 'sum':
                    resampled_dataframe = temp_time_series.resample(frequency).sum()
                if agg == 'mean':
                    resampled_dataframe = temp_time_series.resample(frequency).mean()
                if agg == 'max':
                    resampled_dataframe = temp_time_series.resample(frequency).max()
                if agg == 'min':
                    resampled_dataframe = temp_time_series.resample(frequency).min()
                if agg == 'median':
                    resampled_dataframe = temp_time_series.resample(frequency).median()
                if agg == 'std':
                    resampled_dataframe = temp_time_series.resample(frequency).std()
                if agg == 'var':
                    resampled_dataframe = temp_time_series.resample(frequency).var()
                if agg == 'ffill':
                    resampled_dataframe = temp_time_series.resample(frequency).ffill()
                if agg == 'bfill':
                    resampled_dataframe = temp_time_series.resample(frequency).bfill()
            
            return resampled_dataframe
        
    def to_time_series(self, date_column, value_column, date_format='%Y-%m-%d', window_size=2, one_row=False):
        from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator as SG
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
        
    def dataframe_skip_columns(self, intitial_index, final_index, step=2):
        self.set_dataframe(self.get_dataframe().loc[intitial_index:final_index:step])
        
    def shuffle_dataframe(self):
        self.set_dataframe(self.get_dataframe().sample(frac=1).reset_index(drop=True))
        
    def add_doy_column(self, date_time_column_name='date_time'):
        self.add_column('doy', self.get_column(date_time_column_name).dt.day_of_year)
        
    def add_month_day_column(self, date_time_column_name='date_time'):
        self.add_column('month-day', self.get_column(date_time_column_name).dt.month.astype(str) + '-' + self.get_column(date_time_column_name).dt.day.astype(str))
    
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
        elif scaler_type == 'standard':
            self.__vectorizer = StandardScaler() 
            self.set_dataframe(self.__vectorizer.fit_transform(X=self.get_dataframe()))
        elif scaler_type == 'adjusted_log':
            def log_function(o, min_column):
                return np.log(1 + o - min_column)
            for name in self.get_columns_names():
                min_column = self.get_column(name).min()
                self.transform_column(name, name, log_function, min_column)
        self.convert_dataframe_type()
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
            self.add_column('house_price', y)
            
        elif dataset == 'iris':
            data = load_iris(as_frame=True)
            x = data.data
            y = data.target
            self.set_dataframe(x)
            self.add_column('target', y)  
            
    def similarity_measure_as_column(self, column_name1, column_name2, similarity_method='cosine', weighting_method='tfidf'):
        if similarity_method == 'cosine':
            corpus = self.get_column_as_list(column_name1) + self.get_column_as_list(column_name2)
            vectorizer = Vectorizer(corpus, weighting_method)
            print(len(self.get_column_as_list(column_name1)))
            print(len(self.get_column_as_list(column_name2)))
            new_column = []
            for p in zip(self.get_column_as_list(column_name1), self.get_column_as_list(column_name2)):
                print(p)
                new_column.append(vectorizer.cosine_similarity(p[0], p[1])) 
            
            self.add_column('Similarity score', new_column)
        
        return self.get_dataframe()
        
    @staticmethod
    def outliers_decision_function(o, min_quantile, max_quantile):
        if min_quantile < o < max_quantile:
            return True
        return False
    
    @staticmethod
    def generate_datetime_range(starting_datetime='2013-01-01 00:00:00', end_datetime='2013-12-31 00:00:00', freq='1H', periods=None):
        if periods is not None:
            return pd.date_range(start=starting_datetime, periods=periods, freq=freq)
        return pd.date_range(start=starting_datetime, end=end_datetime, freq=freq)

    @staticmethod
    def scale_trasform_fun(o, max_column):
        return o / max_column