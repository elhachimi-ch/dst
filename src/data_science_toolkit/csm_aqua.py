from .dataframe import DataFrame # from .dataframe import DataFrame in production
from .chart import Chart # from .chart import Chart in production
from .model import Model # from .model import Model in production
from .chart import * # from .chart import * in production
from math import exp
from climatefiller import ClimateFiller
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series
import seaborn as sns
import datetime
import pandas as pd
import numpy as np
import datetime
import time
import warnings
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
from datetime import timedelta
from datetime import timedelta
warnings.filterwarnings('ignore')



class CSM:
    
    def __init__(self, 
                 crop='Wheat', 
                 sowing_date='11/15',
                 start_date_simulation='2015/10/01',
                 end_date_simulation='2016/07/01',
                 weather_data_path='data/r3_dt/r3_aws_full.csv',
                 weather_data_datetime_column_name='datetime',
                 irrigation_method='rainfed'
                 ):
        
        self.start_date_simulation = datetime.datetime.strptime(start_date_simulation, "%Y/%m/%d") 
        self.sowing_date = datetime.datetime(self.start_date_simulation.year, int(sowing_date.split('/')[0]), int(sowing_date.split('/')[1]))
        #data_weather = DataFrame(r"D:\OneDrive - Université Mohammed VI Polytechnique\crsa\data\climate_data_mongo_export.csv")
        self.data_weather = DataFrame(weather_data_path)
        data_p = DataFrame(self.data_weather.get_dataframe(), 'df')
        data_p.resample_timeseries(agg='sum', date_column_name=weather_data_datetime_column_name)
        data_p.keep_columns('p') 

        # data_season_state_monitoring
        self.season_state_monitoring_data = DataFrame()
        
        cf = ClimateFiller(self.data_weather.get_dataframe(), 'df', weather_data_datetime_column_name)
        cf.et0_estimation()

        # Date MinTemp MaxTemp Precipitation ReferenceET

        self.data_weather.set_dataframe(cf.data.get_dataframe())

        self.data_weather.reindex_dataframe(weather_data_datetime_column_name)
        self.data_weather.join(data_p.get_dataframe())
        self.data_weather.index_to_column()
        self.data_weather.rename_columns({weather_data_datetime_column_name: 'Date', 'ta_min': 'MinTemp', 'ta_max': 'MaxTemp', 'p': 'Precipitation', 'et0_pm': 'ReferenceET'})
        self.data_weather.keep_columns(['Date', 'MinTemp', 'MaxTemp', 'Precipitation', 'ReferenceET'])
        
        
        # Date MinTemp MaxTemp Precipitation ReferenceET
        self.data_weather.dataframe = self.data_weather.dataframe[['MinTemp', 'MaxTemp', 'Precipitation', 'ReferenceET', 'Date']]
        
        # Soil    
        soil_r3 = Soil('custom',
                        cn=75,
                        rew=9)
        soil_r3.add_layer(
            soil_r3.zSoil,
            thFC=0.32,
            thWP=0.17,
            thS=0.45,
            Ksat=100,
            penetrability=100
            )
        soil_r3.add_layer_from_texture(
            thickness=soil_r3.zSoil,
            Sand=20,
            Clay=47,
            OrgMat=0,
            penetrability=100
            )
        
        # Crop
        crop_wheat_r3 = Crop(
            crop,
            planting_date=sowing_date,
            # constants
            Tbase=5,
            Tupp=35,
            HI0=0.46,
            WP=16,
            Zmin=0.1,
            Zmax=0.55,
            # variables
            Emergence=82,
            MaxRooting=696,
            Senescence=972,
            Maturity=1462,
            CCx=0.95,
            CGC=0.89,
            CDC=0.6,
            )
        
        
        if irrigation_method == 'rainfed':
            # rainfed
            irrmngt = IrrigationManagement(irrigation_method=0)
        elif irrigation_method == 'et':
            # add net irrigation
            irrmngt = IrrigationManagement(irrigation_method=4)
        elif irrigation_method == 'sm':
            # keep a soil moisture threshold
            irrmngt = IrrigationManagement(irrigation_method=1,SMT=[100]*4)
        
        self.model = AquaCropModel(
                sim_start_time=start_date_simulation,
                sim_end_time=end_date_simulation,
                weather_df=self.data_weather.get_dataframe(),
                soil=soil_r3,
                crop=crop_wheat_r3,
                initial_water_content=InitialWaterContent(value=['FC']),
                irrigation_management=irrmngt,
            )
        """depths = [25] # depth of irrigation applied
        schedule=pd.DataFrame([['2013-11-2'], depths]).T # create pandas DataFrame
        schedule.columns=['Date','Depth'] # name columns
        irrigate_schedule = IrrigationManagement(irrigation_method=3,schedule=schedule)
        self.model.irrigation_management = irrigate_schedule
        print(schedule)"""
        self.model._initialize()
        
    def simulate(self, days=None):
        if days is None:
            self.model.run_model(till_termination=True)
            self.season_results = DataFrame(self.model.get_simulation_results(), 'df')
            self.water_flux_monitoring = DataFrame(self.model.get_water_flux(), 'df')
            self.crop_growth_monitoring = DataFrame(self.model.get_crop_growth(), 'df')
            
            # add date column
            self.water_flux_monitoring.add_column_based_on_function(
                'date',
                lambda row: self.start_date_simulation + timedelta(days=row.name)
            )
            self.water_flux_monitoring.reindex_dataframe('date')
            
            self.crop_growth_monitoring.add_column_based_on_function(
                'date',
                lambda row: self.start_date_simulation + timedelta(days=row.name)
            )
            self.crop_growth_monitoring.reindex_dataframe('date')
            
            self.season_state_monitoring_data = self.water_flux_monitoring.copy()
            self.season_state_monitoring_data.keep_columns(['dap', 'IrrDay', 'ks'])
            #self.crop_growth_monitoring.show()
            #self.crop_growth_monitoring.export('data/r3_dt/season_state.csv')
            self.season_state_monitoring_data.add_column('yield', self.crop_growth_monitoring.get_column('YieldPot'))
            self.season_state_monitoring_data.add_column('yield_actual', self.crop_growth_monitoring.get_column('DryYield'))
            
        else:
            self.model.run_model(days, initialize_model=False)
            water_flux_data = self.model.get_water_flux()
            crop_growth_flux_data = self.model.get_crop_growth()
            if isinstance(water_flux_data, pd.DataFrame):
                data_type = 'df'
                self.water_flux_monitoring = DataFrame(water_flux_data, data_type=data_type)
                self.water_flux_monitoring = DataFrame(water_flux_data, data_type=data_type)
            else:
                data_type = 'matrix'
                self.water_flux_monitoring = DataFrame(
                    water_flux_data,
                    columns_names_as_list=[
                        "time_step_counter",
                        "season_counter",
                        "dap",
                        "Wr",
                        "z_gw",
                        "surface_storage",
                        "IrrDay",
                        "Infl",
                        "Runoff",
                        "DeepPerc",
                        "CR",
                        "GwIn",
                        "Es",
                        "EsPot",
                        "Tr",
                        "TrPot",
                        "ks"
                    ],
                    data_type=data_type
                    )
                
                self.crop_growth_monitoring = DataFrame(
                    crop_growth_flux_data,
                    columns_names_as_list=[
                        "time_step_counter",
                        "season_counter",
                        "dap",
                        "gdd",
                        "gdd_cum",
                        "z_root",
                        "canopy_cover",
                        "canopy_cover_ns",
                        "biomass",
                        "biomass_ns",
                        "harvest_index",
                        "harvest_index_adj",
                        "DryYield",
                        "FreshYield",
                        "YieldPot",
                    ],
                    data_type=data_type
                    )
                
                
    def next_day(self):
        current_date_time = self.model._clock_struct.simulation_start_date + datetime.timedelta(self.model._clock_struct.time_step_counter)
        self.model.run_model(1, initialize_model=False)
        data = self.model.get_water_flux()
        if isinstance(data, pd.DataFrame):
            data_type = 'df'
            self.water_flux_monitoring = DataFrame(data, data_type=data_type)
        else:
            data_type = 'matrix'
            self.water_flux_monitoring = DataFrame(
                data,
                columns_names_as_list=[
                    "time_step_counter",
                    "season_counter",
                    "dap",
                    "Wr",
                    "z_gw",
                    "surface_storage",
                    "IrrDay",
                    "Infl",
                    "Runoff",
                    "DeepPerc",
                    "CR",
                    "GwIn",
                    "Es",
                    "EsPot",
                    "Tr",
                    "TrPot",
                    "ks"
                ],
                data_type=data_type
                )
            
        print(self.water_flux_monitoring.get_row(self.model._clock_struct.time_step_counter))
            
    
    def irrigate(self, amount=0):
        print(self.water_flux_monitoring.show())
        """self.model.irrigation_management
        depths = [25]*len(dates) # depth of irrigation applied
        schedule=pd.DataFrame([dates,depths]).T # create pandas DataFrame
        schedule.columns=['Date','Depth'] # name columns
        irrigate_schedule = IrrigationManagement(irrigation_method=3,schedule=schedule)"""
        
    def monitor(self):
        
        #self.water_flux_monitoring.plot_column('ks')
        
        #print(self.water_flux_monitoring.resample_timeseries('Y', 'sum'))
        self.water_flux_monitoring.show()
        self.crop_growth_monitoring.show()
        self.season_state_monitoring_data.show()
        self.water_flux_monitoring.export('data/r3_dt/water_flux_monitoring.csv')
        self.crop_growth_monitoring.export('data/r3_dt/crop_growth_monitoring.csv')
        self.season_state_monitoring_data.export('data/r3_dt/season_state.csv')
        self.season_results.show()
        #self.water_flux_monitoring.export()

        
        
        