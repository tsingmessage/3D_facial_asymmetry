from collections import defaultdict
import pandas as pd
import numpy as np
from mayavi import mlab
import numpy as np
import pyvista as pv
import cv2
#mlab-三维人脸
#reference:https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fstackoverflow.com%2Fquestions%2F52114655%2Fmayavi-fixed-colorbar-for-all-scenes&amp;data=04%7C01%7Cx.liu.1%40erasmusmc.nl%7Ca12bd176c5604f4240ab08d9feb66fbd%7C526638ba6af34b0fa532a1a511f4ac80%7C0%7C0%7C637820881785970639%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000&amp;sdata=xcYuLxz8fjUQjROn1Ue38jb%2FUr%2FzPb9wvwTnOjRH4Pg%3D&amp;reserved=0
def plot_mlabfaceerror(vertex,dist_error,triangles,landmarker=None, colormap="RdYlBu",reverse_lut=True,
                      opacity=1,azimuth=0, elevation=180,colormap_range=None,colorbar=True,
                      representation='surface',out_file=None,figsize=(1000,1000),data_file=None,color_r=None):
    #mlab.figure(figure="Fittting Error Analysis",fgcolor=(0, 0, 0), bgcolor=(1., 1., 1.))
    mlab.options.offscreen = True
    s=mlab.triangular_mesh(vertex[:,0], vertex[:,1], -vertex[:,2], triangles,
                                colormap=colormap,
                                scalars=dist_error,
                                opacity=opacity,representation=representation)
    s.module_manager.scalar_lut_manager.reverse_lut = reverse_lut
    if colormap_range is not None:
        s.module_manager.scalar_lut_manager.data_range = colormap_range
    if landmarker is not None:
        landmarker=landmarker.reshape(-1,3)
        mlab.points3d(landmarker[:,0], landmarker[:,1], -landmarker[:,2],
                      #color=(1,0,0), mode='sphere', scale_factor=0.015*vertex.max())
                      color=(1,0,0), mode='sphere', scale_factor=10)
    if colorbar:
        mlab.colorbar()
    mlab.view(azimuth=azimuth, elevation=elevation,focalpoint=[ 0, 0, 0])

    #mlab.draw()
    if out_file is not None:
        mlab.savefig(filename=out_file)
    #mlab.show()
    mlab.close()
   #----------------------plot directional heatmap
    mesh1 = pv.read(data_file)
    pv.set_plot_theme("ParaView")
    mesh1['colors'] = -color_r 
    plotter = pv.Plotter(off_screen=True)
    val = colormap_range[1]
    sargs = dict( height=0.25, vertical=True, position_x=0.05, position_y=0.05, color='k')
    plotter.add_mesh(mesh1, scalars='colors', clim=[-val, val],scalar_bar_args=sargs,show_edges=False,rgb=False)
    hsize = 2560
    #plotter.ren_win.OffScreenRenderingOn()
    #plotter.enable_anti_aliasing()
    plotter.ren_win.SetSize([1000, 800])
    #plotter.ren_win.OffScreenRenderingOff()
    #plotter.ren_win.Render()
    plotter.screenshot(out_file, transparent_background=True)
        #plotter.set_background(color = [255/255,193/255,128/255])
    #plotter.plot()
    #cv2.imwrite(out_file.replace('.png','.jpg'), plotter.image)
    #cv2.imwrite(out_file.replace('.png','.jpg'), plotter.image[60:650,250:800,:])
    plotter.clear()


def plot_2mlabvertex(S_vertex,S_colors,S_triangles,
                     F_vertex,F_colors,F_triangles,
                     F_landmarker=None,azimuth=0, elevation=180,
                     representation='wireframe',title=None):
    mlab.figure(figure="Coressponding Face",fgcolor=(0., 0., 0.), bgcolor=(1.0, 1.0, 1.0))

    if S_colors is not None:
        if (S_colors.dtype=='<f4') | (isinstance(S_colors[0,0],float)):
            S_colors=np.hstack(((S_colors*255).astype(np.uint8),255.*np.ones((S_colors.shape[0],1))))
        else:
            S_colors=np.hstack((S_colors.astype(np.uint8),255.*np.ones((S_colors.shape[0],1))))

        mesh = mlab.triangular_mesh(S_vertex[:,0], S_vertex[:,1], -S_vertex[:,2],
                                S_triangles,
                                scalars=np.arange(S_colors.shape[0]),
                                opacity=1,representation='surface')
        mesh.module_manager.scalar_lut_manager.lut.number_of_colors = S_colors.shape[0]
        mesh.module_manager.scalar_lut_manager.lut.table = S_colors
    else:
        mlab.triangular_mesh(S_vertex[:,0], S_vertex[:,1], -S_vertex[:,2], S_triangles,
                                    opacity=1,
                                    colormap="bone",
                                    representation='surface')

    #mlab.triangular_mesh(S_vertex[:,0], S_vertex[:,1], -S_vertex[:,2], S_triangles,
    #                     color=(1, 0, 0),opacity=1,representation='wireframe')
    #plot_vertex(vertices_fitted,std_colors,triangles_fitted,None,1)
    if representation=='wireframe':
        mlab.triangular_mesh(F_vertex[:,0], F_vertex[:,1],-F_vertex[:,2],
                         F_triangles,
                         color=(1, 1, 1),transparent=True,opacity=0.1,representation='wireframe')
    else:
        if F_colors is not None:
            if (F_colors.dtype=='<f4') | (isinstance(F_colors[0,0],float)):
                F_colors=np.hstack(((F_colors*255).astype(np.uint8),255.*np.ones((F_colors.shape[0],1))))
            else:
                F_colors=np.hstack((F_colors.astype(np.uint8),255.*np.ones((F_colors.shape[0],1))))

            mesh = mlab.triangular_mesh(F_vertex[:,0], F_vertex[:,1], -F_vertex[:,2],
                                    F_triangles,
                                    scalars=np.arange(F_colors.shape[0]),
                                    transparent=True,#alpha=0.9,
                                    opacity=0.5,representation='surface')
            mesh.module_manager.scalar_lut_manager.lut.number_of_colors = F_colors.shape[0]
            mesh.module_manager.scalar_lut_manager.lut.table = F_colors
        else:
            mlab.triangular_mesh(F_vertex[:,0], F_vertex[:,1], -F_vertex[:,2], F_triangles,
                                    opacity=1,
                                    colormap="bone",
                                    representation='surface')

    if F_landmarker is not None:
        F_landmarker=F_landmarker.reshape(-1,3)
        mlab.points3d(F_landmarker[:,0], F_landmarker[:,1], -F_landmarker[:,2],
                       color=(1,0,0), mode='sphere', scale_factor=0.015*S_vertex.max())
    mlab.view(azimuth=azimuth, elevation=elevation,focalpoint=[ 0, 0, 0])

    if title is not None:
        mlab.title(title)
    mlab.draw()
    mlab.show()



#sys_byteorder = (">", "<")[sys.byteorder == "little"]

ply_dtypes = dict([
        (b"int8", "i1"),
        (b"char", "i1"),
        (b"uint8", "u1"),
        (b"uchar", "b1"),
        (b"uchar", "u1"),
        (b"int16", "i2"),
        (b"short", "i2"),
        (b"uint16", "u2"),
        (b"ushort", "u2"),
        (b"int32", "i4"),
        (b"int", "i4"),
        (b"uint32", "u4"),
        (b"uint", "u4"),
        (b"float32", "f4"),
        (b"float", "f4"),
        (b"float64", "f8"),
        (b"double", "f8"),
    ]
)

valid_formats = {"ascii": "", "binary_big_endian": ">", "binary_little_endian": "<"}


def read_ply(filename):
    """Read a .ply (binary or ascii) file and store the elements in pandas DataFrame
    Parameters
    ----------
    filename: str
        Path to the filename
    Returns
    -------
    data: dict
        Elements as pandas DataFrames; comments and ob_info as list of string
    """

    with open(filename, "rb") as ply:

        if b"ply" not in ply.readline():
            raise ValueError("The file does not start whith the word ply")
        # get binary_little/big or ascii
        fmt = ply.readline().split()[1].decode()
        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        line = []
        dtypes = defaultdict(list)
        count = 2
        points_size = None
        mesh_size = None
        while b"end_header" not in line and line != b"":
            line = ply.readline()

            if b"element" in line:
                line = line.split()
                name = line[1].decode()
                size = int(line[2])
                if name == "vertex":
                    points_size = size
                elif name == "face":
                    mesh_size = size

            elif b"property" in line:
                line = line.split()
                # element mesh
                if b"list" in line:
                    mesh_names = ["n_points", "v1", "v2", "v3"]

                    if fmt == "ascii":
                        # the first number has different dtype than the list
                        dtypes[name].append((mesh_names[0], ply_dtypes[line[2]]))
                        # rest of the numbers have the same dtype
                        dt = ply_dtypes[line[3]]
                    else:
                        # the first number has different dtype than the list
                        dtypes[name].append((mesh_names[0], ext + ply_dtypes[line[2]]))
                        # rest of the numbers have the same dtype
                        dt = ext + ply_dtypes[line[3]]

                    for j in range(1, 4):
                        dtypes[name].append((mesh_names[j], dt))
                else:
                    if fmt == "ascii":
                        dtypes[name].append((line[2].decode(), ply_dtypes[line[1]]))
                    else:
                        dtypes[name].append(
                            (line[2].decode(), ext + ply_dtypes[line[1]])
                        )
            count += 1

        # for bin
        end_header = ply.tell()

    data = {}

    if fmt == "ascii":
        top = count
        bottom = 0 if mesh_size is None else mesh_size

        names = [x[0] for x in dtypes["vertex"]]
        #print(names)
        data["points"] = pd.read_csv(
            filename,
            sep=" ",
            header=None,
            engine="python",
            skiprows=top,
            skipfooter=bottom,
            usecols=names,
            names=names,
        )

        for n, col in enumerate(data["points"].columns):
            data["points"][col] = data["points"][col].astype(dtypes["vertex"][n][1])

            #print(data["points"])
        #print(mesh_size)
        if mesh_size is not None:
            top = count + points_size

            names = [x[0] for x in dtypes["face"]]
            #usecols = [1, 2, 3]

            data["mesh"] = pd.read_csv(
                filename,
                sep=" ",
                header=None,
                engine="python",
                skiprows=top,
                usecols=names,
                names=names,
            )
            #print(name,data["mesh"] )
            for n, col in enumerate(data["mesh"].columns):
                data["mesh"][col] = data["mesh"][col].astype(dtypes["face"][n][1])

    else:
        with open(filename, "rb") as ply:
            ply.seek(end_header)
            points_np = np.fromfile(ply, dtype=dtypes["vertex"], count=points_size)
            if ext != sys_byteorder:
                points_np = points_np.byteswap().newbyteorder()
            data["points"] = pd.DataFrame(points_np)
            if mesh_size is not None:
                mesh_np = np.fromfile(ply, dtype=dtypes["face"], count=mesh_size)
                if ext != sys_byteorder:
                    mesh_np = mesh_np.byteswap().newbyteorder()
                data["mesh"] = pd.DataFrame(mesh_np)
                data["mesh"].drop("n_points", axis=1, inplace=True)

    vertices=data["points"].values[:,0:3]
    #print(data["points"])
    triangles=data["mesh"].values[:,1:4]

    if data["points"].values.shape[1]>=6:
        colors=data["points"].values[:,3:6]
        if colors.max()>1.: colors=colors/255
    else:
        colors=np.repeat(np.array([30,144,195.]).reshape(1,3),len(vertices),axis=0)/255 #blue
    return vertices,colors,triangles
