from .dataframe import DataFrame # from .dataframe import DataFrame in production
from .chart import Chart # from .chart import Chart in production
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series
import seaborn as sns
import datetime
import pandas as pd


class Crop:
    def _init_(self, crop_type, area, root_depth, max_allowed_depletion):
        self.crop_type = crop_type
        self.area = area
        self.root_depth = root_depth
        self.max_allowed_depletion = max_allowed_depletion
        self.days_growing = 0
        self.water_depletion = 0 

    def get_et_c(self, et_0):
        kc = self.get_kc()
        return kc * et_0

    def get_crop_water_requirement(self, et_0):
        et_c = self.get_et_c(et_0)
        crop_water_req = et_c * self.area * 10
        return crop_water_req

    def get_available_water(self, soil_water_content):
        available_water = (soil_water_content - self.max_allowed_depletion) * self.root_depth * self.area * 10
        return available_water

    def update_water_depletion(self, irrigation_amount, soil_water_content):
        self.water_depletion += irrigation_amount / (self.root_depth * self.area * 10)
        self.water_depletion += (soil_water_content - self.max_allowed_depletion) / (self.root_depth * self.area * 10)

    def simulate_day(self, et_0, irrigation_amount, soil_water_content):
        crop_water_req = self.get_crop_water_requirement(et_0)
        available_water = self.get_available_water(soil_water_content)

        if crop_water_req > available_water:
            irrigation_amount = crop_water_req - available_water
            soil_water_content = self.max_allowed_depletion
        else:
            irrigation_amount = 0
            soil_water_content -= crop_water_req / (self.root_depth * self.area * 10)

        self.update_water_depletion(irrigation_amount, soil_water_content)
        self.days_growing += 1

        return soil_water_content, irrigation_amount

    def simulate_crop(et_0_values, soil_water_content_values, crop):
        for i in range(len(et_0_values)):
            et_0 = et_0_values[i]
            soil_water_content = soil_water_content_values[i]
            soil_water_content, irrigation_amount = crop.simulate_day(et_0, 0, soil_water_content)
            print(f"Day {i+1}: ET0 = {et_0:.2f}, Soil water content = {soil_water_content:.2f}, Irrigation amount = {irrigation_amount:.2f}")

class CSM:
    
    def __init__(self, crop='wheat', area=1, sowing_date_doy=0, season='2013-2014'):
        self.crop = crop
        self.season_length = 242
        self.sowing_date_doy = sowing_date_doy
        # hectare
        self.area = area
        #self.climate_dataframe = self.read_climate_dataframe()
        
        season_years = season.split('-')
        self.monitoring = DataFrame()
        self.monitoring.set_dataframe_index(DataFrame.generate_datetime_range(season_years[0], season_years[1]+'-12-31 23:00', freq='d'))
        self.monitoring.index_to_column()
        self.monitoring.add_month_day_column('index')
        self.monitoring.reindex_dataframe('index')
        self.KC = {
            "initial": 0.2,
            "development": 0.6,
            "mid_season": 1.0,
            "late_season": 0.6
        }
        
        # the duration of a process or the time required to reach a particular stage is expressed in GDD (°C day) instead of number of days.
        # cummulative GDD phases
        self.CGDD_THRESHOLDS = {
            "sowing": 82,
            "development": 0.6,
            "mid_season": 1.0,
            "late_season": 0.6
        }
        
        self.CC_THRESHOLDS = {
            # initial canopy cover at the time of 90% crop emergence [fraction or percentage ground cover]. The initial canopy cover is the product of plant density and the size of
            # the canopy cover per seedling;
            "CC0": 4.5,
            
            # canopy growth coefficient [fraction or percentage ground cover increase per day or growing degree day];
            "CGC": 0.0089,
            #"CGC": 3.5,
            
            # the maximaum canopy cover
            "CCX": 89.33,
            
            # canopy decline coefficient [fraction or percentage ground cover decline per day or growing degree day];
            "CDC": 0.145
        }
        
        
        # The upper temperature threshold specifies the temperature above which crop development no longer increases with an increase in air temperature.
        self.TA_UPPER = 33
        
        # The base temperature (Tbase) is the temperature below which crop development does not progress.
        self.TA_BASE = 5
        
        
        self.CGDD_sowing = 82
        
        
        # crop characteristics #
        # 0.32 m-3/3-3 ou % theta_fc
        self.wc_field_capacity = 0.32
        
        # 0.17 m-3/3-3 ou % theta_fc
        # it is the water quantity below which the crop can no longer extract water, it is the separtion of
        # AW or TAW and NAW
        self.wc_wilting_point = 0.17
        
    def import_climate_data(self, path=r"D:\one_drive\OneDrive - Université Mohammed VI Polytechnique\crsa\data\climate_data_mongo_export.csv", data_type='csv', sheet_name=0, season='2013-2014'):
        # path lab
        #path = "D:\one_drive\OneDrive - Université Mohammed VI Polytechnique\crsa\data\climate_data_mongo_export.csv"
        #path_home = "E:\projects\one_drive\OneDrive - Université Mohammed VI Polytechnique\crsa\data\climate_data_mongo_export.csv"
        climate_data = DataFrame(path)
        climate_data.column_to_date('date_time')
        climate_data.reindex_dataframe('date_time')
        season_years = season.split('-')
        first_year = climate_data.keep_rows_by_year(int(season_years[0]), in_place=False)
        climate_data.keep_rows_by_year(int(season_years[1]))
        climate_data.append_dataframe(first_year)
        temp_climate_data = DataFrame(climate_data.get_dataframe(), data_type='df')
        climate_data.resample_timeseries()
        climate_data.keep_columns(['ws', 'rs', 'doy'])
        climate_data.add_column('ta_min', temp_climate_data.resample_timeseries(agg='min', in_place=False)['ta'])
        climate_data.add_column('ta_max', temp_climate_data.resample_timeseries(agg='max', in_place=False)['ta'])
        climate_data.add_column('rh_min', temp_climate_data.resample_timeseries(agg='min', in_place=False)['rh'])
        climate_data.add_column('rh_max', temp_climate_data.resample_timeseries(agg='max', in_place=False)['rh'])
        climate_data.add_column('p', temp_climate_data.resample_timeseries(agg='sum', in_place=False)['p'])
        climate_data.add_transformed_columns('ta_mean', '(ta_min+ta_max)/2')
        self.monitoring.join(climate_data.get_dataframe(), how='left')
        return 0
    
    def cc_equation1(self, row):
        return self.CC_THRESHOLDS['CC0'] * exp(row['cgdd'] * self.CC_THRESHOLDS['CGC'])
    
    def cc_equation2(self, row):
        return round(self.CC_THRESHOLDS['CCX'] - (0.25 * exp(-row['cgdd'] * self.CC_THRESHOLDS['CGC']) * (self.CC_THRESHOLDS['CCX']**2)/(self.CC_THRESHOLDS['CC0'])), 2)
     
    def cc_equation3(self, row):
        return self.CC_THRESHOLDS['CCX'] * (1 - (0.05 * (exp((3.33 * self.CC_THRESHOLDS['CDC'] * row['cgdd']) / (self.CC_THRESHOLDS['CCX'] + 2.29)) - 1)))
    
    def simulate(self, month_day_sowing_date='11-15', trim=True):
        # 319 349 doy mid november till mid december
        #self.monitoring.add_column("gdd", np.maximum(0, self.monitoring.get_column('ta_mean') - self.TA_BASE)) 
        #self.monitoring.add_column("gdd",  np.minimum(self.monitoring.get_column('gdd'), self.TA_UPPER - self.TA_BASE)) 
        self.monitoring.add_column('gdd', self.monitoring.get_column('ta_mean').clip(lower=self.TA_BASE, upper=self.TA_UPPER) - self.TA_BASE)
        

        # Set GDD values before sowing to 0
        self.monitoring.dataframe.loc[:'2013-'+month_day_sowing_date, 'gdd'] = 0
        
        # Cumulative GDD
        self.monitoring.add_column('cgdd', np.cumsum(self.monitoring.get_column('gdd')))
        
        # Initilise canopy cover time series
        self.monitoring.add_one_value_column('cc', 0)
        
        # Set Canopy Cover before GDD of sowing to 0
        self.monitoring.dataframe.loc[self.monitoring.get_column('cgdd') < self.CGDD_THRESHOLDS['sowing'], 'cc'] = 0
        
        # Set the first value of cc time series to initial threshold
        first_cc_growth_day_index = self.monitoring.dataframe.index[self.monitoring.get_column('cgdd') >= self.CGDD_THRESHOLDS['sowing']].tolist()[0]
        self.monitoring.set_row('cc', first_cc_growth_day_index, self.CC_THRESHOLDS['CC0'])
        
        self.monitoring.add_column_based_on_function('eq1', self.cc_equation1)
        self.monitoring.add_column_based_on_function('eq2', self.cc_equation2)
        
        
        
        eq2_starting_index = self.monitoring.dataframe.index[self.monitoring.get_column('eq1') >= self.CC_THRESHOLDS['CCX']/2].tolist()[0]
        
        # Replace a range of values in df with a range from other_ts
        mask_eq1 = (self.monitoring.dataframe.index >= first_cc_growth_day_index) & (self.monitoring.dataframe.index <= eq2_starting_index)
        self.monitoring.dataframe.loc[mask_eq1, 'cc'] = self.monitoring.dataframe['eq1'][mask_eq1]
        
        
        max_ccx_index = self.monitoring.dataframe.index[self.monitoring.get_column('eq2') >= self.CC_THRESHOLDS['CCX']].tolist()[0]
        self.monitoring.set_row('cgdd', max_ccx_index, 0)
        self.monitoring.add_column('cgdd', np.cumsum(self.monitoring.get_column('gdd')[max_ccx_index:]))
        
        # Replace a range of values in df with a range from other_ts
        mask_eq2 = (self.monitoring.dataframe.index >= eq2_starting_index) & (self.monitoring.dataframe.index <= max_ccx_index)
        self.monitoring.dataframe.loc[mask_eq2, 'cc'] = self.monitoring.dataframe['eq2'][mask_eq2]
        
        
        
        self.monitoring.add_column_based_on_function('eq3', self.cc_equation3)
        
        # Replace a range of values in df with a range from other_ts
        mask_eq3 = (self.monitoring.dataframe.index >= max_ccx_index) & (self.monitoring.dataframe['eq3'] > 0)
        self.monitoring.dataframe.loc[mask_eq3, 'cc'] = self.monitoring.dataframe['eq3'][mask_eq3]
        #self.monitoring.dataframe.loc[self.monitoring.dataframe['cc']<0] = 0
        
        #self.monitoring.dataframe['cc'][max_ccx_index:] = self.monitoring.dataframe['eq3'][max_ccx_index:]
        
        
        #mid_season_index = self.monitoring.dataframe.index[self.monitoring.get_column('eq2') >= self.CC_THRESHOLDS['CCX']].tolist()[0]
        
        
        
        
        
        
        #self.monitoring.transform_column('cc', 'cgdd', self.cc_equation1)
        
        #self.monitoring.transform_column('GDD', 'index', lambda x: x if x > datetime.datetime(2013, 11,5) else 0)
        #self.monitoring.add_column('%CGDD', np.cumsum(self.monitoring.get_column('GDD')))
         
        #df["%GDD"] = df["CGDD"] / total_GDD * 100
        if trim is True:
            start_date = pd.to_datetime('2013-'+month_day_sowing_date)
            end_date = pd.to_datetime('2014-07-01')
            self.monitoring.dataframe = self.monitoring.dataframe.loc[start_date:end_date]
        
        
        

    
    def simulate_fc(self):
        self.monitoring.add_transformed_columns('fc', 'cc/100')
        print(self.monitoring.show())

    def simulate_ndvi(self):
        self.monitoring.add_transformed_columns('ndvi', '(cc/118)+0.14')
    
    def simulate_kcb(self):
        self.monitoring.add_transformed_columns('k_cb', '(1.64*ndvi)-0.2296')

    def simulate_ke(self):
        self.monitoring.add_transformed_columns('k_e', '[0.2*(1−fc)]')

    def simulate_et0(self, method='pm'):
        self.monitoring.add_transformed_columns('et_0', '(1.64*ndvi)-0.2296')

    def simulate_etc(self, method='double'):
        self.monitoring.add_transformed_columns('et_c', '[(1.64 * NDVI)-0.2296]+[0.2 * (1 - fc)]*et_0')

    def simulate_p(self, method='pm'):
        self.monitoring.add_transformed_columns('p', '0.55+0.04*(5-et_c)')

    def simulate_raw(self, method='pm'):
        self.monitoring.add_transformed_columns('raw', '0.55+0.04*(5-et_c)')

    def simulate_taw(self, method='pm'):
        self.monitoring.add_transformed_columns('taw', '1000*(0.32-0.17)*zr')

    def estimate_yield(self, method='last_march_10_days_ndvi'):
        if method == 'max_ndvi':
            ndvi_max = self.monitoring.get_column('ndvi').max()
            estimated_yield = 23.69*ndvi_max - 13.87
        elif method == 'last_march_10_days_ndvi':
            sum_of_last_march_10_days_ndvi = sum(self.monitoring.get_rows(start_index='2014-03-22', end_index='2014-03-31', index_type='datetime', datetime_format='%Y-%m-%d')['ndvi'])
            estimated_yield = 1.79*sum_of_last_march_10_days_ndvi - 8.62
        
        return estimated_yield*self.area
    
    def monitor(self):
        print(self.monitoring.show())
        self.monitoring.plot_column('ndvi', x_label='Date',)
        #print(self.monitoring.get_row('2014-03-20')['ndvi'])
        
        
        #self.monitoring.plot_column('cgdd',)
        #plt.show()
        #self.monitoring.export('data/gdd.csv')
        """fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))
        fig.suptitle('Visual simulation')
        
        # CC
        sns.lineplot(ax=axes[0], x=self.monitoring.get_dataframe().index, y=self.monitoring.get_column('cc').values)
        axes[0].set_title(self.monitoring.get_column('cc').name)
        
        # fc
        #sns.lineplot(ax=axes[1], x=self.monitoring.get_dataframe().index, y=self.monitoring.get_column('fc').values)
        #axes[1].set_title(self.monitoring.get_column('fc').name)
        
        # NDVI
        sns.lineplot(ax=axes[1], x=self.monitoring.get_dataframe().index, y=self.monitoring.get_column('ndvi').values)
        axes[1].set_title(self.monitoring.get_column('ndvi').name)
        plt.show()""" 
        
    
    def read_climate_dataframe(self):
        data = DataFrame('mean_temperature.csv')
        data.keep_columns(['t_mean'])
        return data.get_column_as_list('t_mean') 