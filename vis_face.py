import pyvista as pv
import math
import numpy as np
import glob
from tqdm import tqdm
import cv2
import open3d
import open3d as o3d
from icp import *


def mirror_mesh(mesh):
    mirrored = mesh.copy()
    print(mirrored.points[0,:])
    mirrored.points[:,0] = -mirrored.points[:,0] +100
    print(mirrored.points[0,:])
    return mirrored



path_in = '/media/lau/My Passport/data/Tareq_manuscript_2/transfer_2812465_files_8170e63e/Database_to_Lau/all_mesh/44_post_op_softtissue_rgF.ply'
path_in2 = '/media/lau/My Passport/data/Tareq_manuscript_2/transfer_2812465_files_8170e63e/Database_to_Lau/all_mesh_C_mirrored_aligned/44_post_op_softtissue_rgF_C.ply'

mesh = pv.read(path_in)
#mesh2 = pv.read(path_in2)


mirrored_mesh = mirror_mesh(mesh)  ## Tareq's mirror function

# Convert PyVista mesh to Open3D mesh
source_o3d = o3d.geometry.TriangleMesh()
source_o3d.vertices = o3d.utility.Vector3dVector(mesh.points)
source_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces.reshape(-1, 4)[:, 1:])

mirrored_o3d = o3d.geometry.TriangleMesh()
mirrored_o3d.vertices = o3d.utility.Vector3dVector(mirrored_mesh.points)
mirrored_o3d.triangles = o3d.utility.Vector3iVector(mirrored_mesh.faces.reshape(-1, 4)[:, 1:])


# Perform ICP

P = np.rollaxis(mirrored_mesh.points,1)
X = np.rollaxis(mesh.points,1)
print(P.shape, X.shape)
#exit(0)
Rr, tr, num_iter = IterativeClosestPoint(source_pts = P, target_pts = X, tau = 10e-6)
# Apply transformation to mirrored mesh
#mirrored_o3d.transform(transform)

# transformed new points
Np = ApplyTransformation(P, Rr, tr)
Np = np.rollaxis(Np,1)
print(Np.shape)
#exit(0)
# Convert the transformed Open3D mesh back to a PyVista mesh
#vertices = mirrored_mesh.points
vertices = np.asarray(Np)


faces = np.asarray(mirrored_o3d.triangles)
faces = np.c_[np.full(len(faces), 3), faces]  # Adding a column for the number of vertices per face
mirrored_mesh = pv.PolyData(vertices, faces)

#mesh.plot(color='w', show_edges=True)
#mesh2.plot(color='w', show_edges=True)
# Create a Plotter object
plotter = pv.Plotter(off_screen=False)

#plotter.add_mesh(mirrored_mesh,color="orange",opacity=1)
plotter.add_mesh(mesh,color="white",opacity=1.0)
plotter.set_background('white')
plotter.show()

# Display the plot
plotter.show()



