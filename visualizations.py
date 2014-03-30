import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3dcomponents(Z, labels, axis0, axis1, axis2):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(Z[:,axis0],Z[:,axis1],Z[:,axis2], zdir='z', c=labels)
  plt.show() 
