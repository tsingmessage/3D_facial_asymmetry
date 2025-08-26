import mayavi
from plot_mlabfaceerror import plot_mlabfaceerror, read_ply
from menpo.shape import TriMesh
print('yes')
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import pyvista as pv
import open3d as o3d
import os
from icp import *
from menpo3d.vtkutils import VTKClosestPointLocator

path_out = 'result_out/'

path_input = 'path_to_the_input_ply_mesh/'


def mirror_mesh(mesh):
    mirrored = mesh.copy()
    print(mirrored.points[0,:])
    mirrored.points[:,0] = -mirrored.points[:,0] +100
    print(mirrored.points[0,:])
    return mirrored





def calculate_sum(points1, points2, half_face='left'):

    assert half_face in ['left', 'right'], "half_face must be either 'left' or 'right'"
    if half_face == 'left':
        mask = points1[:, 0] < 0
    else:
        mask = points1[:, 0] > 0

    points1 = points1[mask]
    points2 = points2[mask]


    distances = np.linalg.norm(points1 - points2, axis=1)  # Euclidean distances
    n = distances.shape[0]  # number of points
    return np.sum(distances) / n



ply_file_list = glob.glob(path_input + '*.ply')
imageid_list, score_list,score_list_up,score_list_mid,score_list_low,score_list_up_mid = [], [],[],[],[],[]

count_all = 0

df_index = pd.read_csv('facial_segment_index/segment_index.csv')
print(df_index.head())
index_up  = np.array(df_index['up'].values)[0:976].astype(int)-1
index_mid  = np.array(df_index['mid'].values)[0:2042].astype(int)-1
index_low  = np.array(df_index['low'].values)[0:2005].astype(int)-1
index_up_mid = np.concatenate((index_up, index_mid))

for i_total, data_file in tqdm(enumerate(ply_file_list)):

    file_out = data_file.replace(path_input,'').replace('.ply','')
    imageid_list = np.append(imageid_list, str(file_out))


    mesh = pv.read(data_file)

    # Mirror the mesh
    mirrored_mesh = mirror_mesh(mesh)  ## 
    ###

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
    print(vertices)
    #exit(0)
    faces = np.asarray(mirrored_o3d.triangles)
    faces = np.c_[np.full(len(faces), 3), faces]  # Adding a column for the number of vertices per face
    mirrored_mesh = pv.PolyData(vertices, faces)

    # Compute the difference between the original and mirrored mesh
    closest_points_locator = VTKClosestPointLocator(mirrored_mesh)
    closest_points, closest_idx = closest_points_locator(mesh.points)

    # Calculate the sum
    mfa_all = calculate_sum(mesh.points, closest_points)
    mfa_up = calculate_sum(mesh.points[index_up], closest_points[index_up])
    mfa_mid = calculate_sum(mesh.points[index_mid], closest_points[index_mid])
    mfa_up_mid = calculate_sum(mesh.points[index_up_mid], closest_points[index_up_mid])
    print("Asymmetry Value: ", mfa_all)
    #exit(0)
    count_all = count_all + 1
    #if count_all > 160:
    #    break

    score_list = np.append(score_list, mfa_all)
    score_list_up = np.append(score_list_up, mfa_up)
    score_list_mid = np.append(score_list_mid, mfa_mid)
    score_list_up_mid = np.append(score_list_up_mid, mfa_up_mid)



#plot_mlabfaceerror(shape_tmp,dist_error_sum/count_all,face_tmp,colormap='autumn', colormap_range=[0,10],out_file = 'avg_cases.png')
data_df = pd.DataFrame({ 'imageID': imageid_list,  'MFA_all':score_list,'MFA_upper_face':score_list_up,'MFA_middle_face':score_list_mid, 'MFA_upper_middle':score_list_up_mid})
data_df.to_csv(path_out+'summary_rgF_C_final_tareq_pipeline.csv',index=None)
