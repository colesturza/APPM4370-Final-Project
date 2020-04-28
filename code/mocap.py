from modules.FORCE import Force
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate

N = 1500
p = 0.2
g = 1.5
alpha = 1.0
dt = 0.1

path = '../mocap_data/'

data = pd.read_csv(path + 'walk_subj_2_legs.csv')

# rolling average to smooth out data
data = data.rolling(15, min_periods=1).mean()

# Convert to numpy ndarray
data = data.to_numpy()

# Store number of readouts
num_readouts = data.shape[1]

# Old simulation time
simtime = data.shape[0]
steps = np.arange(0, simtime, 1)

# New simulation time, with smaller steps
new_steps = np.arange(0, simtime-1, dt)
steps_predict = np.arange(simtime-1, simtime+50, dt)
new_simtime = len(new_steps)

# Create interpolation function for new points
f = interpolate.interp1d(x=steps, y=data, axis=0, kind='cubic')

new_data = f(new_steps)

rnn = Force(N=N, p=p, g=g, readouts=num_readouts)

zt, _ = rnn.fit(new_steps, new_data)

zpt = rnn.predict(steps_predict)

print(zt.shape)

# Approximation using leading PCs
lw_f, lw_z = 3.5, 1.5
fig1, ax_fz = plt.subplots()
fig1.set_tight_layout(True)
sns.set_style('white')
sns.despine()
ax_fz.plot(new_steps, new_data[:,17], lw=lw_z, label='interp', color='firebrick')
ax_fz.plot(new_steps, zt[:,17], lw=lw_z, label='interp', color='blue')

fig2, ax2 = plt.subplots()
fig2.set_tight_layout(True)
sns.set_style('white')
sns.despine()

ax2.plot(steps_predict, zpt[:,17], lw=lw_z, label='interp', color='blue')

plt.show()
