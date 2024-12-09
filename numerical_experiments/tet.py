import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from random import randint

#tetrahedron
W = np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]])/np.sqrt(3)

#facets
hull = ConvexHull(W)
true_facets = hull.simplices
true_facets.shape

#color dict
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

fig = plt.figure(figsize=(10, 10))

r = 0.05
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
x = np.cos(u) * np.sin(v) *0.99
y = np.sin(u) * np.sin(v)*0.99
z = np.cos(v)*0.99

ax = fig.add_subplot(projection='3d')
ax.plot_surface(x, y, z, cmap = plt.cm.bone, alpha=0.1)



for facet in true_facets:
    
    point1, point2, point3 = W[facet[0]], W[facet[1]], W[facet[2]]
    
    vertices = [point1, point2, point3]
    
    ax.add_collection3d(Poly3DCollection([vertices], alpha=0.85,color = color_dict[tuple(np.sort(facet))]))
    
    lines = [[point1,point2],[point2,point3],[point3,point1]]
    
    for line in lines:
        line = np.array(line)
        ax.plot(line[:,0],line[:,1],line[:,2],color="darkblue")


plt.grid(False)
plt.axis('off')
ax.text(0, -1.5, 2.2, "Facets of the inscribing polytope", color='black')
plt.show()