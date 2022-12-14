import pandas as pd
import numpy as np

filename = 'simplecorridor_centerline.csv'
data = pd.read_csv(filename, usecols=['x', 'y'])
data = np.array(data)