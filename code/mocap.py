from modules.FORCE import Force
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '../../mocap_data'

df = pd.read_csv(path + '').to_numpy()
