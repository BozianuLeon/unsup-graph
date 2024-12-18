

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cylinder parameters
radius = 24
length = 72
num_points = 12
num_poles = 5

# Evenly distribute points along the height (z-axis)
# z = np.linspace(0, length, num_points)
z = np.random.uniform(0, length, num_points*num_poles)
print(z.shape)
print(np.repeat(z, num_poles).shape)
# Evenly distribute points around the circle in the xy-plane
theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

# Calculate x and y coordinates for each point
x = radius * np.cos(theta)
y = radius * np.sin(theta)
print(x.shape)

# Stack the coordinates
# We will duplicate the (x, y) for each height value in z
# pole_points = np.vstack([np.tile(x, num_poles), np.tile(y, num_poles), np.repeat(z, num_poles)]).T
pole_points = np.vstack([np.tile(x, num_poles), np.tile(y, num_poles), z]).T

num_lid_points = 72
# Set z coordinate to 0 for all points (circle at z=0)
lid_theta = np.random.uniform(0, 2 * np.pi, num_lid_points)
# Generate uniformly distributed radial distances (inside the circle)
r = np.sqrt(np.random.uniform(0, radius**2, num_lid_points))
# Convert polar coordinates to Cartesian coordinates
lid_x = r * np.cos(lid_theta)
lid_y = r * np.sin(lid_theta)
lid_z = np.zeros(num_lid_points)
lid_points = np.vstack([lid_x, lid_y, lid_z]).T

lid_theta2 = np.random.uniform(0, 2 * np.pi, num_lid_points)
r2 = np.sqrt(np.random.uniform(0, radius**2, num_lid_points))
lid_points2 = np.vstack([r2 * np.cos(lid_theta2), r2 * np.sin(lid_theta2), length*np.ones(num_lid_points)]).T


# points = np.vstack([pole_points,lid_points])
points = pole_points





# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
ax.scatter(lid_points[:, 0], lid_points[:, 1], lid_points[:, 2], c='b', marker='o')
ax.scatter(lid_points2[:, 0], lid_points2[:, 1], lid_points2[:, 2], c='g', marker='o')

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim((-5,80))
plt.show()
