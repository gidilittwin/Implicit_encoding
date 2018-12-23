import numpy as np

import plotly as py
import plotly.graph_objs as go


def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

#skeleton = np.array([[0, 1, 2, 3, 4],[0, 5, 6, 7, 8],[0, 9, 10, 11, 12],[0, 13, 14, 15, 16],[0, 17, 18, 19, 20]])
#skeleton2 = np.array([[0, 1, 2, 3, 4,5],[0, 6,7,8,9,10],[0, 11,12,13,14,15],[0, 16,17,18,19,20],[0,21,22,23,24,25]])

skeleton21 = [np.array([0, 1, 2, 3, 4]),np.array([0, 5, 6, 7, 8]),np.array([0, 9, 10, 11, 12]),np.array([0, 13, 14, 15, 16]),np.array([0, 17, 18, 19, 20])]
skeleton22 = [np.array([0, 1, 2, 3, 4]),np.array([0, 5, 6, 7, 8]),np.array([0, 9, 10, 11, 12]),np.array([0, 13, 14, 15, 16]),np.array([0, 17, 18, 19, 20]),np.array([0, 21])]






def mesh_plot(obj,idx=0,type_='mesh'):
    

    traces = []
    vertices = obj[idx]['vertices']
    vertices_up = obj[idx]['vertices_up']
    faces    = obj[idx]['faces']

    
    # Grid limits
    traces.append(go.Scatter3d(x=[-1,-1,-1,-1,1,1,1,1],
                               y=[-1,-1,1,1,-1,-1,1,1],
                               z=[-1,1,-1,1,-1,1,-1,1],
                               mode='markers',
                               opacity=0.0,
                               marker=dict(size=0.0,opacity=0.0 ))   ) 
        
    # Mesh
    if type_=='mesh' or type_=='cubed':
        traces.append(go.Mesh3d(x=vertices[:,0],
                                y=vertices[:,1],
                                z=vertices[:,2],
#                                colorscale = [['0'  , 'rgba(20,29,67,0.6)'], 
#                                              ['0.5', 'rgba(51,255,255 ,0.6)'], 
#                                              ['1'  , 'rgba(255  ,191,0,0.6)']],                                           
                                intensity = vertices[:,2]*255,
                                opacity=1.0,
                                i=faces[:,0],
                                j=faces[:,1],
                                k=faces[:,2]))

    elif type_=='cloud':
        colors   = obj[idx]['field']

        # Markers
        traces.append(go.Scatter3d(x=vertices[:,0],
                                    y=vertices[:,1],
                                    z=vertices[:,2],
                                    mode='markers',
                                    marker=dict(size=4,colorscale='Viridis',color=colors[:,0]*255,opacity=0.6 ))   ) 






    elif type_=='cloud_up':
        # Markers
        traces.append(go.Scatter3d(x=vertices_up[:,0],
                                    y=vertices_up[:,1],
                                    z=vertices_up[:,2],
                                    mode='markers',
                                    marker=dict(size=2,line=dict(color='rgba(217, 217, 217, 0.8)',width=0.5),opacity=0.8 ))   ) 
        




#
#    # Projection
#    colorscale=[['0'  , 'rgba(20,29,67,0.6)'], 
#                ['0.5', 'rgba(51,255,255 ,0.6)'], 
#                ['1'  , 'rgba(255  ,191,0,0.6)']]
#    voxels_= hand_model_['reprojected']
#    voxels_= voxels_[idx,:,:,0]*255
#    vshape = voxels_.shape[1]
#    Y,X = np.meshgrid(np.linspace(0, 1, vshape),np.linspace(0, 1, vshape))
#    traces.append(go.Surface(z=0*np.ones(voxels_.shape),
#                    x=X,
#                    y=Y,
#                    colorscale=colorscale,
#                    showlegend=False,
#                    showscale=False,
#                    surfacecolor=voxels_))



    layout = dict(
        width=1200,
        height=1200,
        autosize=False,
        title='Mesh',
        scene=dict(
            xaxis=dict(range=[-1, 1],
                gridcolor='rgb(255, 255, 255)',
#                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'),
            yaxis=dict(range=[-1, 1],
                gridcolor='rgb(255, 255, 255)',
#                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'),
            zaxis=dict(range=[-1, 1],
                gridcolor='rgb(255, 255, 255)',
#                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(28,76,96)')
            ),)


    fig = dict(data=traces, layout=layout)
#    py.offline.plot(fig,auto_open=True, filename='/Users/gidilittwin/Desktop/plotly_'+type_+'_'+str(idx)+'.html')
    py.offline.plot(fig)
    
    
    
    
def double_mesh_plot(obj,idx=0,type_='mesh'):
    

    traces = []


    
    # Grid limits
    traces.append(go.Scatter3d(x=[-1,-1,-1,-1,1,1,1,1],
                               y=[-1,-1,1,1,-1,-1,1,1],
                               z=[-1,1,-1,1,-1,1,-1,1],
                               mode='markers',
                               opacity=0.0,
                               marker=dict(size=0.0,opacity=0.0 ))   ) 
        
    # Mesh
    vertices = obj[0]['vertices']
    faces    = obj[0]['faces']    
    traces.append(go.Mesh3d(x=vertices[:,0],
                            y=vertices[:,1],
                            z=vertices[:,2],
#                            colorscale = [['0'  , 'rgba(20,29,67,0.6)'], 
#                                          ['0.5', 'rgba(51,255,255 ,0.6)'], 
#                                          ['1'  , 'rgba(255  ,191,0,0.6)']],                                           
                            intensity = vertices[:,2]*255,
                            opacity=1.0,
                            i=faces[:,0],
                            j=faces[:,1],
                            k=faces[:,2]))

    vertices = obj[1]['vertices']
    faces    = obj[1]['faces']    
    traces.append(go.Mesh3d(x=vertices[:,0],
                            y=vertices[:,1],
                            z=vertices[:,2],
#                            colorscale = [['0'  , 'rgba(255,45,25 ,0.6)'], 
#                                          ['0.5', 'rgba(255,45,25 ,0.6)'], 
#                                          ['1'  , 'rgba(255,45,25 ,0.6)']],                                           
                            intensity = vertices[:,2]*110,
                            opacity=0.2,
                            i=faces[:,0],
                            j=faces[:,1],
                            k=faces[:,2]))
 

    layout = dict(
        width=1200,
        height=1200,
        autosize=False,
        title='Mesh',
        scene=dict(
            xaxis=dict(range=[-1.1, 1.1],
                gridcolor='rgb(255, 255, 255)',
#                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'),
            yaxis=dict(range=[-1.1, 1.1],
                gridcolor='rgb(255, 255, 255)',
#                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'),
            zaxis=dict(range=[-1.1, 1.1],
                gridcolor='rgb(255, 255, 255)',
#                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(28,76,96)')
            ),)


    fig = dict(data=traces, layout=layout)
#    py.offline.plot(fig,auto_open=True, filename='/Users/gidilittwin/Desktop/plotly_'+type_+'_'+str(idx)+'.html')
    py.offline.plot(fig)
    
    
    

def cloud_plot(pointcloud,cloud_gt,reprojected,idx=0):
    
    traces = []
    pointcloud = (pointcloud+1)/2
    cloud_gt = (cloud_gt+1)/2
        
        
    # BONES GT
    for i in np.arange(0,len(skeleton21)):
        bones = skeleton21[i]
        for j in np.arange(1,bones.shape[0]):
            bone = cloud_gt[idx,bones[j-1:j+1],:]
            traces.append(go.Scatter3d(x=bone[:,0],
                                       y=bone[:,1],
                                       z=bone[:,2],
                                       mode='lines',
                                       line=dict(color='rgba(217, 217, 217, 0.14)',width=5)))
          
            
    # Markers
    traces.append(go.Scatter3d(x=pointcloud[idx,:,0],
                                y=pointcloud[idx,:,1],
                                z=pointcloud[idx,:,2],
                                mode='markers',
                                marker=dict(size=2,line=dict(color='rgba(217, 217, 217, 0.8)',width=0.5),opacity=0.8 ))   ) 
    
    # Projection
    colorscale=[['0'  , 'rgba(20,29,67,0.6)'], 
              ['0.5', 'rgba(51,255,255 ,0.6)'], 
              ['1'  , 'rgba(255  ,191,0,0.6)']]
    voxels_= reprojected
    vshape = voxels_.shape[1]
    voxels_= voxels_[idx,:,:,0]*255
    Y,X = np.meshgrid(np.linspace(0, 1, vshape),np.linspace(0, 1, vshape))
    traces.append(go.Surface(z=np.zeros(voxels_.shape),
                    x=X,
                    y=Y,
                    colorscale=colorscale,
                    showlegend=False,
                    showscale=False,
                    surfacecolor=voxels_))
        
    
    
    
    
    
    # Grid limits
    traces.append(go.Scatter3d(x=[0,0,0,0,1,1,1,1],
                               y=[0,0,1,1,0,0,1,1],
                               z=[0,1,0,1,0,1,0,1],
                               mode='markers',
                               marker=dict(size=0.0,opacity=1.0 ))   ) 
    
    
    layout = dict(
        width=1600,
        height=1200,
        autosize=False,
        title='PROCESSED_INPUT_DATA',
        scene=dict(
            xaxis=dict(range=[0, 1],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'),
            yaxis=dict(range=[0, 1],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'),
            zaxis=dict(range=[0, 1],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(28,76,96)')
            ),)


    fig = dict(data=traces, layout=layout)
    py.offline.plot(fig,auto_open=False, filename='/Users/gidilittwin/Desktop/plotly_input_data'+str(idx)+'.html')
#    py.offline.plot(fig)
    
    
        
def data_cloud_plot(pointcloud):
    
    traces = []
    pointcloud = (pointcloud+1)/2
              
            
    # Markers
    traces.append(go.Scatter3d(x=pointcloud[:,0],
                                y=pointcloud[:,1],
                                z=pointcloud[:,2],
                                mode='markers',
                                marker=dict(size=2,line=dict(color='rgba(217, 217, 217, 0.14)',width=0.5),opacity=0.8 ))   ) 
    # Grid limits
    traces.append(go.Scatter3d(x=[0,0,0,0,1,1,1,1],
                               y=[0,0,1,1,0,0,1,1],
                               z=[0,1,0,1,0,1,0,1],
                               mode='markers',
                               marker=dict(size=0.0,opacity=1.0 ))   ) 
    
    
    layout = dict(
        width=1200,
        height=800,
        autosize=False,
        title='Mesh',
        scene=dict(
            xaxis=dict(range=[0, 1],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'),
            yaxis=dict(range=[0, 1],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'),
            zaxis=dict(range=[0, 1],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(28,76,96)')
            ),)


    fig = dict(data=traces, layout=layout)
    py.offline.plot(fig)
        