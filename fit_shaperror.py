from matplotlib import cm#,colors
from menpo3d.vtkutils import trimesh_to_vtk, VTKClosestPointLocator
from menpo.transform import Translation, UniformScale, AlignmentSimilarity
import numpy as np 
def fit_shaperror(source,target,flag_Near_vertices=True,flag_direction=False):
    #lm_align = AlignmentSimilarity(source.landmarks[group],
    #                           target.landmarks[group]).as_non_alignment()
    #source = lm_align.apply(source)
    points = source.points.copy()#trilist,source.trilist.copy()
    
    #if group is not None:
    #    landmarker=source.landmarks[group].points.copy()
        
    tr = Translation(-1 * source.centre())
    sc = UniformScale(1.0 / np.sqrt(np.sum(source.range() ** 2)), 3)
    prepare = tr.compose_before(sc)

    source = prepare.apply(source)
    target = prepare.apply(target)
    # store how to undo the similarity transform
    restore = prepare.pseudoinverse()
    
    target_vtk = trimesh_to_vtk(target)
    closest_points_on_target = VTKClosestPointLocator(target_vtk)

    #target_tri_normals = target.tri_normals()
    #
    U, tri_indices = closest_points_on_target(source.points.copy())
    Near_vertices=restore.apply(U)
    print(Near_vertices, points)
    
    #-------------------------------------------------------directional heatmap-------
    index_half = np.where(points[:,0]<0)
    center_point = np.mean(points,axis=0)
    center_point = center_point - [0, 15, 55] 
    distance0 = points - center_point
    distance0 = np.sqrt(distance0[:,0]*distance0[:,0] + distance0[:,1]*distance0[:,1] +distance0[:,2]*distance0[:,2])
   
    distance1 = Near_vertices - center_point
    distance1 = np.sqrt(distance1[:,0]*distance1[:,0] + distance1[:,1]*distance1[:,1] +distance1[:,2]*distance1[:,2])
    color_r = distance1 - distance0
    color_r[index_half] = 0

    #-------------------------------------------end directional heatmap--------------------

    #fface.plot_2mlabvertex(vertices,colors,triangles,
    #                   Near_vertices,S_colors,F_triangles,
    #                   F_landmarker)
    #fface.plot_mlabvertex(Near_vertices,S_colors,F_triangles,F_landmarker) 

    dist_SF0=np.sqrt(np.sum((points-Near_vertices)**2,1))
    #dist_SF0[index_half] = 0
    #pal_color=cm.get_cmap( 'seismic',101)(np.linspace(0, 1, 101))*255
    min_depth=np.min(dist_SF0)
    max_depth=np.max(dist_SF0)
    print("Max Error:",max_depth," Min Error:",min_depth," Mean Error:",dist_SF0.mean())
    #dist_SF=(dist_SF0-min_depth)/(max_depth-min_depth)*100
    #print(dist_SF)
    #detal_color=pal_color[dist_SF.astype(int),0:3].astype(np.uint8)
    #detal_color=np.hstack((detal_color,255.*np.ones((len(detal_color),1))))  # NX4 
#    if flag_show:
#        if group is not None:
#            plot_mlabfaceerror(points,dist_SF,trilist,landmarker)
#        else:
#            plot_mlabfaceerror(points,dist_SF,trilist)
    
    if flag_direction:
        # center=points.mean(0).reshape(1,3)
        # dist_target=np.sum((points-center)**2,1)
        # dist_source=np.sum((Near_vertices-center)**2,1)
        # dist_SF0[dist_target>dist_source]=-dist_SF0[dist_target>dist_source]
            
        bc=source.vertex_normals()#get_normal(input_head,triangles)
        ba=points-Near_vertices
        angles=np.array([line2angle(x,y)  for x,y in zip(ba,bc)])
        dist_SF0[angles<90]=-dist_SF0[angles<90]

    if flag_Near_vertices:
        return color_r, dist_SF0,Near_vertices,tri_indices , dist_SF0.mean()
    else:
        return dist_SF0
