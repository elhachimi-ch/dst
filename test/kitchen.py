import time
import os
import sys
sys.path.insert(0, os.path.abspath('../src/'))
from src.dataframe import DataFrame
from model import Model
import dst as dst

def main():
    ti = time.time()
    data = DataFrame()
    data.load_dataset('iris')
    y =  data.get_column('target')
    data.drop_column('target')
    
    # decision tree model
    model = Model(data_x=data.get_dataframe(), data_y=y, model_type='dt', training_percent=0.8)
    
    # train the model
    model.train()
    
    # get all classification evaluation metrics
    model.report()
    
    #get the cross validation
    model.cross_validation(5)
   
    print(time.time() - ti)


if __name__ == '__main__':
    main()
