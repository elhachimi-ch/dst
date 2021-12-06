import numpy as np
import time
from numpy.linalg import matrix_power
from nltk.corpus import stopwords
from dataframe import DataFrame
from vectorizer import Vectorizer, stemming
from model import Model
from chart import Chart
import pandas as pd
from lib import *

def efficiency(o):
    return o/1.501

def main():
    ti = time.time()
    
    print(time.time() - ti)
    
if __name__ == '__main__':
    main()
