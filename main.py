from fit_shaperror import fit_shaperror 
import mayavi
from plot_mlabfaceerror import plot_mlabfaceerror, read_ply
from menpo.shape import TriMesh
print('yes')
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np

#120529112309 120625091023 120621112333 ---------------------161209111758---200911110000-140627105925

#120504102419 
#120508131642 
#120509113006
#140627105925
#140919103136
#160715121428
#path1='/media/lau/My Passport/data/facial_asymmetry/dense_FA_aligned/'
#path2='/media/lau/My Passport/data/facial_asymmetry/dense_FA_aligned_mirror_aligned/'
path_out = 'result_out/'
path2 = 'dense_derma_aligned/'
path1= 'dense_derma_aligned_mirror_aligned/'


df_index = pd.read_csv('facial_segment_index/segment_index.csv')
print(df_index.head())
index_up  = np.array(df_index['up'].values)[0:976].astype(int)-1
index_mid  = np.array(df_index['mid'].values)[0:2042].astype(int)-1
index_low  = np.array(df_index['low'].values)[0:2005].astype(int)-1
#print(index_up)

ply_file_list = glob.glob(path2 + '*.ply')
imageid_list, score_list,score_list_up,score_list_mid,score_list_low = [], [],[],[],[]
dist_error_sum = np.zeros([5023])
count_all = 0
for i_total, data_file in tqdm(enumerate(ply_file_list)):

    #data_file = path1+'160715121428.ply'
    print(data_file)
    face_shp1,face_col1,face_tri1=read_ply(data_file)
    #print(face_shp1)
    face_shp2,face_col2,face_tri2=read_ply(data_file.replace(path2,path1))
    #print(face_shp2)
    file_out = data_file.replace(path2,'').replace('.ply','')


    imageid_list = np.append(imageid_list, str(file_out).zfill(12))


    file_out = path_out + file_out
    #plot_2mlabvertex(face_shp1,face_col1,face_tri1,face_shp2,face_col2,face_tri2)


    target=TriMesh(face_shp2,face_tri2)
    source=TriMesh(face_shp1,face_tri1)    
    directional_hetmap_color, dist_error,N_vertices,tri_indices ,mean_err =fit_shaperror(source,target,flag_Near_vertices=True)
    #print(dist_error.shape, np.mean(dist_error[index_up]), np.mean(dist_error[index_mid]),np.mean(dist_error[index_low]),mean_err)
    #exit(0)
    dist_error_sum = dist_error_sum + dist_error
    count_all = count_all + 1
    #if count_all > 160:
    #    break


    score_list = np.append(score_list, mean_err)
    score_list_up = np.append(score_list_up, np.mean(dist_error[index_up]))
    score_list_mid = np.append(score_list_mid, np.mean(dist_error[index_mid]))
    score_list_low = np.append(score_list_low, np.mean(dist_error[index_low]))
    file_out = file_out + '_' + str(mean_err) + '.png'
    plot_mlabfaceerror(face_shp2,dist_error,face_tri2,colormap='autumn',colormap_range=[0,5],\
                       out_file = file_out, data_file=data_file, color_r=directional_hetmap_color)
    #exit(0)


print(count_all)
print(dist_error_sum/count_all)
#shape_tmp,col_tmp,face_tmp=read_ply(path_template)
#plot_mlabfaceerror(shape_tmp,dist_error_sum/count_all,face_tmp,colormap='autumn', colormap_range=[0,10],out_file = 'avg_cases.png')
data_df = pd.DataFrame({ 'imageID': imageid_list,  'score_all':score_list,'score_upper_face':score_list_up,'score_middle_face':score_list_mid,'score_lower_face':score_list_low})
data_df.to_csv(path_out+'summary.csv',index=None)
