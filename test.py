# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 20:11:15 2022

@author: Owner
"""

import numpy as np
from Odometry import OdometryClass
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

frames = 'C:/Users/Owner/OneDrive/Documents/BU/CS585/Challenge/Challenge2/Challenge/video_train'
test_odometry = OdometryClass(frames)
path, gt = test_odometry.run()
#np.save('predictions',path)

#MSE calculation
mse_scores= []
for i in range(len(path)):
    mse_scores.append(np.linalg.norm(path[i] - gt[i]))
    
mse_scores = np.mean(mse_scores)
print('Average MSE score:', mse_scores)

#Plot the stuff
fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(path[:,0],path[:,1],path[:,2],'r',label="Estimated")
ax.plot3D(gt[:,0],gt[:,1],gt[:,2],'b',label="GT")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
fig.show()

# for angle in range(0,360):
#     ax.view_init(30,angle)
#     plt.draw()
#     plt.pause(.001)