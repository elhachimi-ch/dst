from .gis import GIS # from .gis import GIS in production
from .csm_aqua import CSM # from .csm_aqua import CSM in production
import imp
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from dataframe import DataFrame
import math
from dataframe import DataFrame
import random
import re
import geopandas as gpd
import contextily as cx
   

class R3(gym.Env): 
    
    # 
    MAX_SOWING_DAY = 100
    
    # action space definition
    ADD_DAY = 0
    SUB_DAY = 1
    SAME_DAY = 2
    action_space = gym.spaces.Discrete(2)
    actions = [ADD_DAY, SUB_DAY]
    
    # observation space definition
    observation_space = gym.spaces.Box(low = 0, 
                                            high = 100,
                                            shape = (33,),
                                            dtype = int)
    AGENT = -1
    
    def __init__(self, stochastic=True, fitness_threshold=-1000):
        self.r3_plots_path = "plots.shp"
        self.r3_pipelines_path = "pipelines.shp"
        self.layers = GIS()
        self.layers.add_data_layer('plots', self.r3_plots_path)
        self.layers.add_data_layer('pipelines', self.r3_pipelines_path)
        self.csm = CSM()
        self.sow()
        self.fitness_threshold = fitness_threshold
        
    def sow(self, sowing_dates_series=None):
        if sowing_dates_series is None:
            self.layers.add_random_series_column('pipelines', 'sowing_dates')
            #self.layers.add_random_series_column('plots', 'sowing_dates')
        else:
            self.layers.add_column('pipelines', sowing_dates_series, 'sowing_dates')
            #self.layers.add_column('plots', sowing_dates_series, 'sowing_dates')
    
    def get_state(self):
        """start_state = np.where(self.grid_state == self.AGENT)
        start_not_found = not (start_state[0] and goal_state[0])
        if start_not_found:
            print("Start state not present in the Gridworld. Check the Grid layout")

        #start_state = (start_state[0][0])
        start_state = 0"""
        
        return self.layers.get_column('sowing_dates')
    
    
    def step(self, action):
        """
        Run one step into the env
        Args:
            state (Any): Current index state of the maze
            action (int): Discrete action for up, down,
            left, right
            slip (bool, optional): Stochasticity in the 
            env. Defaults to True.
        Raises:
            ValueError: If invalid action is provided as 
            input
        Returns:
            Tuple : Next state, reward, done, _
       
            return next observation, reward, done, info
        """
        
        action = int(action)
        info = {"success": True}
        self.grid_state, reward = self.get_state_reward(action)
        done = (reward > self.fitness_threshold)
        if done is False:
            reward = 1.0
        else:
            reward = -2.0
        return self.grid_state, reward, done, info
    
    def get_state_reward(self, action):
        actual_fitness = self.fitness_sowing_dates_distribution()
        random_plot = np.random.randint(0, 33)
        actual_sowing_dates = self.layers.get_column('pipelines', 'sowing_dates').to_numpy()
        #print(actual_sowing_dates)
        if action == self.ADD_DAY:
            if actual_sowing_dates[random_plot] <= 100: 
                actual_sowing_dates[random_plot] += 1
        elif action == self.SUB_DAY:
            if actual_sowing_dates[random_plot] != 0:
                actual_sowing_dates[random_plot] += -1
        elif action == self.SAME_DAY:
            pass
        #print("Taken action:", action)
        self.sow(actual_sowing_dates)
        
        #self.layers.set_row('pipelines', 'sowing_dates', random_plot, actual_sowing_dates)
        self.estimated_cluster_yield()
        reward = self.fitness_sowing_dates_distribution() - actual_fitness
        return self.layers.get_column('pipelines', 'sowing_dates').to_numpy(), reward
    
    def render(self):
        return self.show()
    
    def reset(self):
        self.sow()
        return self.layers.get_column('pipelines', 'sowing_dates').to_numpy()
        #return self.set_sowing_dates(np.ones((116, 1)))
    
    def get_sowing_dates(self):
        return self.layers.get_column('sowing_dates')
    
    def get_delta_list(self, delta=7):
        network_constraints_matrix = []
        for index_q, row_q in self.layers.get_data_layer('pipelines').iterrows():
            network_constraints_list = []
            for index_p, row_p in self.layers.get_data_layer('pipelines').iterrows():
                if row_p['sowing_dates'] >= row_q['sowing_dates'] - delta and row_p['sowing_dates'] <= row_q['sowing_dates'] + delta:
                    network_constraints_list.append(row_p['canal_id'])
            network_constraints_matrix.append(network_constraints_list)
        return network_constraints_matrix 
    
    def estimated_cluster_yield(self):
        yield_series = []
        for p in self.layers.get_column('pipelines', 'sowing_dates'):
            self.csm.simulate_fc(p)
            self.csm.simulate_ndvi()
            yield_series.append(self.csm.estimate_yield()/100)
        self.layers.add_column('pipelines', yield_series, 'yield')
        return sum(self.layers.get_column('pipelines', 'yield'))
    
    def shows(self, plot=False):
        ax = self.layers.plot(layer_name='pipelines', column4color='canal_id', alpha=0.8)
        #self.layers.show_dataframe(layer_name='plots')
        #self.layers.show_dataframe(layer_name='pipelines')
        print(self.layers.get_data_layer('pipelines'))
        if plot is True:
            # Add numbers to the plot
            for i, row in self.layers.get_data_layer('plots').iterrows():
                centroid = row['geometry'].centroid
                ax.annotate(row['area'], xy=(centroid.x, centroid.y), ha='center', va='center', fontsize=3)
                ax.text(centroid.x, centroid.y, str(row['area']), ha='center', va='center', fontsize=3)
                
            self.layers.show() 
            
    def show(self):
        self.fig, self.ax = plt.subplots(figsize=(17,17))
        
        # Sample GeoDataFrame with MultiLineString geometry
        gdf = gpd.read_file(self.r3_plots_path)
        
        converted_gdf = gdf.to_crs(epsg=3857)
        
        self.ax = converted_gdf.plot(ax=self.ax, alpha=0.5, column='canal_id')
        
        cx.add_basemap(ax=self.ax, source=cx.providers.Esri.WorldImagery)
        import pandas  as pd
        series = pd.Series(np.random.randint(0, 50, converted_gdf.shape[0]))
        converted_gdf['sowing_dates'] = series
        
        
        # Add numbers to the plot
        for _, row in converted_gdf.iterrows():
            centroid = row['geometry'].centroid
            #ax.text(centroid.x, centroid.y, str(row['Numbers']), ha='center', va='center', fontsize=10)
            self.ax.annotate(row['sowing_dates'], xy=(centroid.x, centroid.y), xytext=(3, 3),
                        textcoords='offset points', ha='center', va='center', fontsize=9, color='yellow')

        # Set plot title and axis labels
        self.ax.set_title('A spatiotemporal distribution of sowing dates')
        
        plt.show()

    
    def fitness_sowing_dates_distribution(self):
        self.verify_irrigation_network_constraints()
        return self.estimated_cluster_yield() - sum(self.layers.get_column('pipelines', 'remaining'))
        
    def verify_irrigation_network_constraints(self):
        remaining_list = []
        delta_list = self.get_delta_list()
        for list_voisin in delta_list:
            list_score = 0
            for canal in list_voisin:
                splited_canal = re.findall('[A-Z]*\d+', canal)
                next_branch = ""
                total_remaining = 0
                for p in splited_canal:
                    temp_canals_list = list_voisin
                    temp_canals_list = list(map(lambda x: re.sub('-', '', x), temp_canals_list))
                    remaining = 0
                    next_branch += p
                    if self.layers.get_data_layer('pipelines')[self.layers.get_column('pipelines', 'canal_id_i') == next_branch]['capacity'].shape[0] > 0:
                        #print('verify ', next_branch)
                        if next_branch in temp_canals_list:
                            temp_canals_list.remove(next_branch)
                        common_canal_capacity = self.layers.get_data_layer('pipelines')[self.layers.get_column('pipelines', 'canal_id_i') == next_branch].capacity.to_numpy()[0]
                        activated_canals_sum = 0
                        for q in temp_canals_list:
                            if next_branch in q:
                                activated_canals_sum += self.layers.get_data_layer('pipelines')[self.layers.get_column('pipelines', 'canal_id_i') == q].capacity.to_numpy()[0]
                        
                        remaining = common_canal_capacity - activated_canals_sum
                        total_remaining += remaining
                list_score += total_remaining
                #print("List score:", list_score) 
            
            remaining_list.append(list_score)
            self.layers.add_column('pipelines', remaining_list, 'remaining')
            #print("Total remaining:", total_remaining)