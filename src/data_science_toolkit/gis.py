import numpy as np
import time
import sys
import math
from numpy.linalg import matrix_power
import nltk
import lib
from .dataframe import DataFrame
from .vectorizer import Vectorizer, stemming
from .model import Model
from .rl import *
from .chart import Chart
import pandas as pd
from .lib import *
import contextily as cx
from matplotlib import pyplot as plt
import geopandas as gpd
import folium as fl
from shapely.geometry import Point
import xarray as xr
import cfgrib

class GIS:
    """
    GIS class
    """
    def __init__(self):
        self.data_layers = {}
        self.fig, self.ax = plt.subplots(figsize=(17,17))

    def add_data_layer(self, layer_path, layer_name, data_source='sf'):
        if data_source == 'df':
            self.data_layers[layer_name] = gpd.GeoDataFrame(layer_path)
        elif data_source == 'sf':
            self.data_layers[layer_name] = gpd.read_file(layer_path)
        
    def get_data_layer(self, layer_name):
        return self.data_layers.get(layer_name)
    
    def join_layer(self, layer_name, geo_datframe_to_join, on):
        self.data_layers[layer_name] = self.data_layers.get(layer_name).merge(geo_datframe_to_join, on=on)
        
    def plot(self, layer_name, column4color=None, color=None, alpha=0.5, legend=False, 
             figsize_tuple=(15,10), cmap=None, ):
        """_summary_

        Args:
            layer_name (_type_): _description_
            column4color (_type_, optional): _description_. Defaults to None.
            color (_type_, optional): _description_. Defaults to None.
            alpha (float, optional): _description_. Defaults to 0.5.
            legend (bool, optional): _description_. Defaults to False.
            figsize_tuple (tuple, optional): _description_. Defaults to (15,10).
            cmap (str, optional): exmaple: 'Reds' for heatmaps. Defaults to None.
        """
        layer = self.data_layers.get(layer_name).to_crs(epsg=3857)
        layer.plot(ax=self.ax, alpha=alpha, edgecolor='k', color=color, legend=legend, 
                   column=column4color, figsize=figsize_tuple, cmap=cmap)
        cx.add_basemap(ax=self.ax, source=cx.providers.Esri.WorldImagery)
        
    def show(self, layer_name=None, interactive_mode=False):
        if interactive_mode is True: 
            return self.data_layers.get(layer_name).explore()
        else:
            self.ax.set_aspect('equal')
            plt.show()
        
    def get_crs(self, layer_name):
        """
        Cordonate Reference System
        EPSG: european petroleum survey group
        """
        return self.get_data_layer(layer_name).crs
    
    def export(self, layer_name, file_name, file_format='geojson'):
        if file_format == 'geojson':
            self.data_layers[layer_name].to_file(file_name + '.geojson', driver='GeoJSON')
        elif file_format == 'shapefile':
            self.data_layers[layer_name].to_file(file_name + '.shp')
            
    def to_crs(self, layer_name, epsg="3857"):
        self.data = self.data_layers[layer_name].to_crs(epsg)
        
    def set_crs(self, layer_name, epsg="3857"):
        self.data = self.data_layers[layer_name].set_crs(epsg)
        
    def show_points(self, x_y_csv_path, crs="3857"):
        pass
    
    def show_point(self, x_y_tuple, crs="3857"):
        pass
    
    def add_point(self, x_y_tuple, layer_name, crs="3857"):
        point = Point(0.0, 0.0)
        #self.__dataframe = self.get_dataframe().append(row_as_dict, ignore_index=True)
        row_as_dict = {'geometry': point}
        self.data_layers[layer_name].append(row_as_dict, ignore_index=True)
    
    def new_data_layer(self, layer_name, crs="EPSG:3857"):
        self.data_layers[layer_name] = gpd.GeoDataFrame(crs=crs)
        self.data_layers[layer_name].crs = crs
        
    def add_column(self, layer_name, column, column_name):
        y = column
        if (not isinstance(column, pd.core.series.Series or not isinstance(column, pd.core.frame.DataFrame))):
            y = np.array(column)
            y = np.reshape(y, (y.shape[0],))
            y = pd.Series(y)
        self.data_layers[layer_name][column_name] = y
        
    def show_data_layer(self, layer_name, number_of_row=None):
        if number_of_row is None:
            print(self.get_data_layer(layer_name))
        elif number_of_row < 0:
            return self.get_data_layer(layer_name).tail(abs(number_of_row)) 
        else:
            return self.get_data_layer(layer_name).head(number_of_row) 
        
    def add_row(self, layer_name, row_as_dict):
        self.data_layers[layer_name] = self.get_data_layer(layer_name).append(row_as_dict, ignore_index=True)
    
    def get_row(self, layer_name, row_index, column=None):
        if column is not None:
            return self.data_layers[layer_name].loc[self.data_layers[layer_name][column] == row_index].reset_index(drop=True)
        return self.data_layers[layer_name].iloc[row_index]
    
    def get_layer_shape(self, layer_name):
        """
        return (Number of lines, number of columns)
        """
        return self.data_layers[layer_name].shape
    
    def get_columns_names(self, layer_name):
        header = list(self.data_layers[layer_name].columns)
        return header 
    
    def drop_column(self, layer_name, column_name):
        """Drop a given column from the dataframe given its name

        Args:
            column (str): name of the column to drop

        Returns:
            [dataframe]: the dataframe with the column dropped
        """
        self.data_layers[layer_name] = self.data_layers[layer_name].drop(column_name, axis=1)
        return self.data_layers[layer_name]
    
    def keep_columns(self, layer_name, columns_names_as_list):
        for p in self.get_columns_names(layer_name):
            if p not in columns_names_as_list:
                self.data_layers[layer_name] = self.data_layers[layer_name].drop(p, axis=1)
                
    def get_area_column(self, layer_name):
        return self.get_data_layer(layer_name).area
    
    def get_row_area(self, layer_name, row_index):
        return self.data_layers[layer_name].area.iloc[row_index]
    
    def get_distance(self, layer_name, index_column, row_index_a, row_index_b):
        if 1 == 1:
            other = self.get_row(layer_name, row_index_b, index_column)
            return self.get_row(layer_name, row_index_a, index_column).distance(other)
    
    def filter_dataframe(self, layer_name, column, func_de_decision, in_place=True, *args):
        if in_place is True:
            if len(args) == 2:
                self.set_dataframe(
                    self.data_layers[layer_name].loc[self.get_column(column).apply(func_de_decision, args=(args[0], args[1]))])
            else:
                self.set_dataframe(
                    self.data_layers[layer_name].loc[self.get_column(column).apply(func_de_decision)])
        else:
            if len(args) == 2:
                return self.data_layers[layer_name].loc[self.get_column(column).apply(func_de_decision, args=(args[0], args[1]))]
            else:
                return self.data_layers[layer_name].loc[self.get_column(column).apply(func_de_decision)]

    def transform_column(self, column_to_trsform, column_src, fun_de_trasformation, *args):
        if (len(args) != 0):
            self.set_column(column_to_trsform, self.get_column(column_src).apply(fun_de_trasformation, args=(args[0],)))
        else:
            self.set_column(column_to_trsform, self.get_column(column_src).apply(fun_de_trasformation))
    
    def get_column(self, layer_name, column):
        return self.data_layers[layer_name][column]
    
    def reindex_dataframe(self, layer_name, index_as_liste=None, index_as_column_name=None):
        if index_as_liste is not None:
            new_index = new_index = index_as_liste
            self.data_layers[layer_name].index = new_index
        if index_as_column_name is not None:
            self.data_layers[layer_name].set_index(index_as_column_name, inplace=True)
        if index_as_column_name is None and index_as_liste is None:
            new_index = pd.Series(np.arange(self.get_shape()[0]))
            self.data_layers[layer_name].index = new_index
            
    def get_era5_land_grib_as_dataframe(self, file_path, layer_name):
        grip_path = file_path
        ds = xr.load_dataset(grip_path, engine="cfgrib")
        self.data_layers[layer_name] = DataFrame()
        self.data_layers[layer_name].set_dataframe(ds.to_dataframe())
        return ds.to_dataframe()
    
    def rename_columns(self, layer_name, column_dict_or_all_list, all_columns=False):
        if all_columns is True:
            types = {}
            self.data_layers[layer_name].columns = column_dict_or_all_list
            for p in column_dict_or_all_list:
                types[p] = str
            self.data_layers[layer_name] = self.get_dataframe().astype(types)
        else:
            self.data_layers[layer_name].rename(columns=column_dict_or_all_list, inplace=True)
            
    def calculate_area_as_column(self, layer_name):
        self.add_column(layer_name, self.get_area_column(layer_name), 'area')
         
    
    @staticmethod
    def new_geodaraframe_from_points():
        map.new_data_layer('valves', crs="ESRI:102191")
        for p in range(map.get_layer_shape('pipelines')[0]):
            #print(map.get_row('pipelines', p))
            vi = Point(map.get_row('pipelines', p)['X_Start'], map.get_row('pipelines', p)['Y_Start'])
            vf = Point(map.get_row('pipelines', p)['X_End'], map.get_row('pipelines', p)['Y_End'])
            id_pipeline = map.get_row('pipelines', p)['Nom_CANAL']
            row_as_dict = {'id_pipeline': id_pipeline,
                        'geometry': vi}
            map.add_row('valves', row_as_dict)
            row_as_dict = {'id_pipeline': id_pipeline,
                        'geometry': vf}
            map.add_row('valves', row_as_dict)
        
    
        
    