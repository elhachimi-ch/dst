import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd



class Chart:
    chart_type_list = ['line', 'bar', 'box', 'swarm', 'strip_swarm', 'count', 'scatter', 'dist', 'point', 'pair',
                  'correlation_map', 'reg', 'heat_map']

    def __init__(self, dataframe=None, column4x=None, chart_type='pair', group_by=None, columns_names_list=None, plotly=False):
        self.dataframe = dataframe
        if column4x is None:
            self.column4x = dataframe.index
        else:
            self.column4x = column4x
        self.chart_type = chart_type
        self.group_by = group_by
        self.columns_names_list = columns_names_list
        self.plotly = plotly
        sns.set_theme(color_codes=True)
        

    def add_data_to_show(self, data_column=None, column4hover=None, column4size=None, y_column=None, color=None):
        print(self.chart_type)
        if self.plotly == True:
            if self.chart_type == self.chart_type_list[0]:
                self.fig = px.line(self.dataframe, x=self.column4x, y=data_column, color=self.group_by, hover_name=column4hover)
            elif self.chart_type == self.chart_type_list[1]:
                self.ax = sns.barplot(data=self.dataframe, x=self.column4x, y=data_column, hue=self.group_by)
                loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
                self.ax.xaxis.set_major_locator(loc)
            elif self.chart_type == self.chart_type_list[2]:
                sns.boxplot(data=self.dataframe, x=self.column4x, y=data_column, hue=self.group_by)
            elif self.chart_type == self.chart_type_list[3]:
                sns.swarmplot(data=self.dataframe, x=self.column4x, y=data_column, hue=self.group_by)
            elif self.chart_type == self.chart_type_list[4]:
                sns.stripplot(data=self.dataframe, x=self.column4x, y=data_column, hue=self.group_by)
            elif self.chart_type == self.chart_type_list[5]:
                sns.countplot(data=self.dataframe, x=self.column4x, y=data_column, hue=self.group_by)
            elif self.chart_type == self.chart_type_list[6]:
                sns.regplot(data=self.dataframe, x=self.column4x, y=data_column, hue=self.group_by)
            elif self.chart_type == self.chart_type_list[7]:
                sns.distplot(self.dataframe.column, kde=False, hue=self.group_by)
            elif self.chart_type == self.chart_type_list[8]:
                sns.pointplot(x="class", y="survived", hue=self.group_by, data=self.dataframe, palette={"male": "g", "female": "m"},
                            markers=["^", "o"], linestyles=["-", "--"], capsize=.2)
            elif self.chart_type == self.chart_type_list[9]:
                dataframe = DataFrame()
                dataframe.set_data_frame(self.dataframe)
                self.fig = px.scatter_matrix(self.dataframe, dimensions=dataframe.get_columns_names(),
                    color=self.group_by)
            elif self.chart_type == self.chart_type_list[10]:
                sns.clustermap(self.dataframe.corr(), cmap=sns.diverging_palette(230, 20, as_cmap=True), annot=True,
                            fmt='1%',
                            center=0.0)
            elif self.chart_type == self.chart_type_list[11]:
                self.fig = px.scatter(self.dataframe, x=self.column4x, y=data_column, color=self.group_by, size=column4size, hover_name=column4hover)
        else:
            #sns.set_style("whitegrid")
            if self.chart_type == self.chart_type_list[0]:
                self.ax = sns.lineplot(data=self.dataframe, x=self.column4x, y=data_column, markers=True, hue=self.group_by)
            elif self.chart_type == self.chart_type_list[1]:
                sns.set_theme(style="whitegrid")
                self.ax = sns.barplot(data=self.dataframe, x=self.column4x, y=data_column, hue=self.group_by)
            elif self.chart_type == self.chart_type_list[2]:
                self.ax = sns.boxplot(data=self.dataframe, x=self.column4x, y=data_column, hue=self.group_by)
            elif self.chart_type == self.chart_type_list[3]:
                self.ax = sns.swarmplot(data=self.dataframe, x=self.column4x, y=data_column, hue=self.group_by)
            elif self.chart_type == self.chart_type_list[4]:
                self.ax = sns.stripplot(data=self.dataframe, x=self.column4x, y=data_column, hue=self.group_by)
            elif self.chart_type == self.chart_type_list[5]:
                self.ax = sns.countplot(data=self.dataframe, x=self.column4x, y=data_column, hue=self.group_by)
            elif self.chart_type == self.chart_type_list[6]:
                self.ax = sns.regplot(data=self.dataframe, x=self.column4x, y=data_column, hue=self.group_by)
            elif self.chart_type == self.chart_type_list[7]:
                self.ax = sns.distplot(self.dataframe.column, kde=False, hue=self.group_by)
            elif self.chart_type == self.chart_type_list[8]:
                self.ax = sns.pointplot(x="class", y="survived", hue=self.group_by, data=self.dataframe, palette={"male": "g", "female": "m"},
                            markers=["^", "o"], linestyles=["-", "--"], capsize=.2)
            elif self.chart_type == self.chart_type_list[9]:
                self.ax = sns.pairplot(self.dataframe, hue=self.group_by, vars=self.columns_names_list)
            elif self.chart_type == self.chart_type_list[10]:
                self.ax = sns.clustermap(self.dataframe.corr(), annot=True, center=0.0)
            elif self.chart_type == self.chart_type_list[11]:
                self.ax = sns.jointplot(x=data_column, y=y_column, data=self.dataframe, kind="reg", color=color)
                #self.ax = sns.scatterplot(data=self.dataframe, x=data_column, y=y_column)
            elif self.chart_type == self.chart_type_list[12]:
                # Compute the correlation matrix
                corr = self.dataframe.corr()
                # Generate a mask for the upper triangle
                mask = np.triu(np.ones_like(corr, dtype=bool))
                # Generate a custom diverging colormap
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                # Draw the heatmap with the mask and correct aspect ratio
                self.ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                            square=True, linewidths=.5, cbar_kws={"shrink": .5})
            return self.ax
    
    def plot_on_map(self,
                    iso_locations_column=None,
                    circle_size_column=None,
                    animation_frame_column=None,
                    hover_name_column=None, 
                    projection='natural earth',
                    scope='world'):
        self.fig = px.scatter_geo(
            self.dataframe,
            locations=iso_locations_column,
            size=circle_size_column,
            animation_frame=animation_frame_column,
            hover_name=hover_name_column,
            color=self.group_by,
            projection=projection,
            scope=scope,
            )
        
    def plot_colored_map(self,
                    iso_locations_column=None,
                    color_column=None,
                    animation_frame_column="Year",
                    scope='world',
                    hover_name_column=None, # column to add to hover information
                    ):# column on which to animate):
       self.fig = px.choropleth(
                    self.dataframe,
                    locations=iso_locations_column,
                    scope=scope,
                    color=color_column, # lifeExp is a column of gapminder
                    hover_name=hover_name_column, # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Plasma,
                    animation_frame=animation_frame_column,
                    projection='natural earth'# column on which to animate
                    )
                
        
    def show(self):
        if self.plotly:
            self.fig.show()
        else:
            plt.show()

    def config(self,
               title="", 
               x_label="X", 
               y_label="Y", 
               x_limit_i=None, 
               x_limit_f=None, 
               y_limit_i=None, 
               y_limit_f=None, 
               interval=None,
               x_rotation_angle=90,
               y_rotation_angle=0,
               titile_font_size=29,
               x_label_font_size=13,
               y_label_font_size=13,
               x_font_size=11,
               y_font_size=11,
               ):
        if self.plotly:
            self.fig.update_layout(
            # add a title text for the plot
            title_text = title,
            # set projection style for the plot
            #geo = dict(projection={'type':'natural earth'}
            ) # by default, projection type is set to 'equirectangular'
        else:
            plt.title(title)
            plt.xlim(x_limit_i, x_limit_f)
            plt.ylim(y_limit_i, y_limit_f)
            plt.xticks(rotation=x_rotation_angle, fontsize=x_font_size)
            plt.yticks(rotation=y_rotation_angle, fontsize=y_font_size)
            self.ax.set_title(title,fontsize=titile_font_size)
            self.ax.set_xlabel(x_label,fontsize=x_label_font_size)
            self.ax.set_ylabel(y_label,fontsize=y_label_font_size)
            
            if interval is not None:
                loc = plticker.MultipleLocator(base=interval) # this locator puts ticks at regular intervals
                self.ax.xaxis.set_major_locator(loc)


    def save(self, chart_path="output.png", transparent=False):
        if self.plotly is True:
            self.fig.savefig(chart_path, transparent=transparent, bbox_inches = 'tight', dpi=600)
        else:
            self.ax.savefig(chart_path, transparent=transparent, bbox_inches = 'tight', dpi=600)
