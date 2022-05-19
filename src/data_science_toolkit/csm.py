from math import exp
from dataframe import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series
from chart import Chart
import seaborn as sns

class CSM:
    
    def __init__(self):
        self.crop_type = 'Wheat'
        self.season_length = 242
        self.climate_dataframe = self.read_climate_dataframe()
        self.monitoring = DataFrame(Series([i+1 for i in range(self.season_length)]), ['day'], data_type='list')
        
        self.CC0 = 4.5
        self.CCx = 89.33
        self.CGC = 0.0089
        self.CDC = 0.145
        self.CCx_2 = self.CCx / 2
        self.Tupper = 33
        self.Tbase = 5
        self.CGDD_sowing = 82
        
        # hectare
        self.area = 100
        
        
        # crop characteristics #
        # 0.32 m-3/3-3 ou % theta_fc
        self.wc_field_capacity = 0.32
        
        # 0.17 m-3/3-3 ou % theta_fc
        # it is the water quantity below which the crop can no longer extract water, it is the separtion of
        # AW or TAW and NAW
        self.wc_wilting_point = 0.17
        
    def cc_equation1(self, t):
        return self.CC0 * exp(t * self.CGC)
    
    def cc_equation2(self, t):
        return self.CCx - (0.25 * exp(-t * self.CGC) * (self.CCx**2)/(self.CC0))
     
    def cc_equation3(self, t):
        return self.CCx * (1 - (0.05 * (exp((3.33 * self.CDC) * t/(self.CCx + 2.29)) - 1))) 
    
    def simulate_canopy_cover(self, offset=0):
        Tmb = np.zeros((self.season_length,))
        ti = np.zeros((self.season_length,))
        CC = np.zeros((self.season_length,))
        Eq1 = np.zeros((self.season_length,))
        Eq2 = np.zeros((self.season_length,))
        Eq3 = np.zeros((self.season_length,))
        for day in range(offset, self.season_length):
                if (self.climate_dataframe[day] < 5):
                    Tmb[day] = 0 
                else:
                    if (self.climate_dataframe[day] >= self.Tupper):
                        Tmb[day] = self.Tupper - self.Tbase
                    else:
                        Tmb[day] = self.climate_dataframe[day] - self.Tbase
        
        ti[offset] = Tmb[offset]
        for k in range((offset + 1), 242):
            ti[k] = Tmb[k] + ti[k - 1]

        t0_all = np.argwhere(ti >= self.CGDD_sowing)
        t0 = t0_all[0]
        

        for i in range(offset, t0[0]):
            CC[i] = 0

        CC[t0[0]] = self.CC0
        ti[t0[0]] = 0
 
        
        for p in range((t0[0] + 1), 242):
            ti[p] = Tmb[p] + ti[p - 1]

        for m in range((t0[0] + 1), 242):
            Eq1[m] = self.cc_equation1(ti[m])
            
        
        for m in range((t0[0] + 1),242):
            Eq1[m] = self.cc_equation1(ti[m])
            Eq2[m] = self.cc_equation2(ti[m])
            Eq2[m] = Eq2[m].round(2)

        p1 = np.argwhere(Eq1 >= self.CCx_2)
        phase1 = p1[0][0]

        for ii in range((t0[0] + 1), phase1):
            CC[ii] = Eq1[ii]

        p2 = np.argwhere(Eq2 >= self.CCx)
        phase2 = p2[0][0]
        
        for jj in range(phase1, phase2):
            CC[jj] = Eq2[jj]

        ti[phase2] = 0
        CC[phase2] = self.CCx

        for kk in range((phase2 + 1), 242):
            ti[kk] = Tmb[kk] + ti[kk - 1]
            Eq3[kk] = self.cc_equation3(ti[kk])
            if (Eq3[kk] >= 0):
                CC[kk] = Eq3[kk]
            else:
                CC[kk] = 0
        
        for kk in range((phase2 + 1), 242):
            if (Eq3[kk] < self.CCx_2):
                day_final = kk - 1
                break
            
        self.monitoring.add_column(CC, 'cc')   
        return CC
    
    def simulate_fc(self):
        self.monitoring.add_transformed_columns('fc', 'cc/100')

    def simulate_ndvi(self):
        self.monitoring.add_transformed_columns('ndvi', '(cc/118)+0.14')
    
    def simulate_kcb(self):
        self.monitoring.add_transformed_columns('k_cb', '(1.64*ndvi)-0.2296')

    def simulate_ke(self):
        self.monitoring.add_transformed_columns('k_e', '[0.2 (1−fc)]')

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

    def estimate_yield(self, method='last_10_ndvi'):
        ndvi_list = self.monitoring.get_column('ndvi')
        if method == 'max_ndvi':
            ndvi_max = float(max(ndvi_list))
            estimated_yield = 23.69*ndvi_max - 13.87
        elif method == 'last_10_ndvi':
            ndvi_list = list(ndvi_list)
            sum_of_last_10_ndvi = sum([float(ndvi_list[153-i]) for i in range(10)])
            estimated_yield = 1.79*sum_of_last_10_ndvi - 8.62
        
        return estimated_yield*self.area
    
    def monitor(self):
        self.monitoring.show()
        fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))
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
        plt.show() 
        
    
    def read_climate_dataframe(self):
        data = DataFrame('mean_temperature.csv')
        data.keep_columns(['t_mean'])
        return data.get_column_as_list('t_mean')