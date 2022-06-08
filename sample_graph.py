import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

f = h5py.File("depths.mat")
mpl.use('tkagg')


X = np.array([range(0, 480)])
Y = np.array([range(0, 640)])
X, Y = np.meshgrid(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=10., azim=60)
ax.plot_surface(X, Y, f["depths"][5], cmap=cm.coolwarm)
plt.show()