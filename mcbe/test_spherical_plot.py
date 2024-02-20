
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import mcbe

from PIL import Image
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from random import randint
import random

num_vert = 30
W = mcbe.random_sphere(num_vert,3)[0]


#calculate true facets
hull = ConvexHull(W)
true_facets = hull.simplices
true_facets.shape

#create color dict
color = []
true_facets_set = set([tuple(np.sort(x)) for x in true_facets])
all_facets = true_facets_set
n = len(all_facets)

for i in range(n):
    color.append('#%06X' % randint(0, 0xFFFFFF))
    
color_dict = {}
ind_color = 0

for facet in all_facets:
    color_dict[facet] =  color[ind_color]
    ind_color = ind_color +1







num_points = 200
# Generate spherical coordinates
theta = np.linspace(0, 2 * np.pi, num_points)
phi = np.linspace(0, np.pi, num_points)

# Create a meshgrid from spherical coordinates
theta, phi = np.meshgrid(theta, phi)

# Parametric equations for a sphere
x =  np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

pattern = pd.DataFrame(np.zeros((num_points,num_points)))


for i in tqdm(range(num_points)):
    for j in range(num_points):
        point = np.array([x[i,j],y[i,j],z[i,j]])

        

        corr_x_vert = [np.dot(point, i) for i in W]
        #find subframes
        subframe = tuple(np.sort(np.argsort(corr_x_vert)[-3:]))
        if subframe in color_dict.keys():
            pattern.iloc[i,j] = color_dict[subframe]
            
        else:
            # Generate random hexadecimal color code
            red = random.randint(0, 255)
            green = random.randint(0, 255)
            blue = random.randint(0, 255)

            color_code = f"#{red:02x}{green:02x}{blue:02x}"
            
            color_dict[subframe] = color_code
            pattern.iloc[i,j] = color_code


for color in color_dict.values():



    # Create 3D plot
                
    fig = plt.figure(figsize=(10, 10))

    r = 0.05
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x_sphere = np.cos(u) * np.sin(v) *0.99
    y_sphere = np.sin(u) * np.sin(v)*0.99
    z_sphere = np.cos(v)*0.99

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_sphere, y_sphere, z_sphere, cmap = plt.cm.bone, alpha=0.1)

    subframe = np.array(list(color_dict.keys()))[np.array(list(color_dict.values())) == color][0]
    print(subframe)

    # Plot the surface with the pattern
    
    
    #enumerate vertices
    for i in range(len(W)):
        ax.text(W[i,0],W[i,1],W[i,2],str(i),color='black',fontsize=12,fontweight='bold')
    ax.scatter(np.array(x).flatten()[np.array(list(np.array(pattern).flatten())) == color],np.array(y).flatten()[np.array(list(np.array(pattern).flatten())) == color],np.array(z).flatten()[np.array(list(np.array(pattern).flatten())) == color],color = color,alpha=0.2)
    
    ax.scatter(W[:,0],W[:,1],W[:,2],color='black',s=100,alpha=0.5)

    ax.scatter(W[subframe[0],0],W[subframe[0],1],W[subframe[0],2],color='red',s=100)
    ax.scatter(W[subframe[1],0],W[subframe[1],1],W[subframe[1],2],color='red',s=100)
    ax.scatter(W[subframe[2],0],W[subframe[2],1],W[subframe[2],2],color='red',s=100)


    ax.text(0, -1.25, 2.2, "Sampling-based sub-frames", color='black')

    # Remove axes
    ax.set_axis_off()

    #set xlim and ylim
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])



    # Show the plot
    plt.show()

