import pyvista as pv
import numpy as np
import csv


def get_label(filename, factor):
    
    retarr = np.zeros((0,7))
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            items = row[0].split(',')
            retarr = np.insert(retarr,0,np.asarray(items),0)
    
    
    retarr[:,0:6] = retarr[:,0:6] * factor
    
    return retarr



def disp_model(filename):
    

    pv.set_plot_theme("document")
    mesh = pv.PolyData(filename+'.STL')
    
    
    plotter = pv.Plotter()
    
    plotter.add_mesh(mesh,opacity=0.8,color='#FFFFFF')
    
   
    shapetypes = ['O ring', 'Through hole', 'Blind hole', 'Triangular passage', 'Rectangular passage', 'Circular through slot', 'Triangular through slot', 'Rectangular through slot', 'Rectangular blind slot','Triangular pocket', 'Rectangular pocket', 'Circular end pocket', 'Triangular blind step', 'Circular blind step', 'Rectangular blind step', 'Rectangular through step' , '2-sides through step', 'Slanted through step', 'Chamfer', 'Round', 'Vertical circular end blind slot', 'Horizontal circular end blind slot', '6-sides passage', '6-sides pocket']

    
    colors = ['#000080','#FF0000','#FFFF00','#00BFFF','#DC143C','#DAA520','#DDA0DD','#708090','#556B2F','#483D8B','#CD5C5C','#21618C','#1C2833','#4169E1','#1E90FF','#FFD700','#FF4500','#646464','#DC143C','#98FB98','#9370DB','#8B4513','#00FF00','#008080']
    
    items = get_label(filename+'.csv', 1000)
    
    flag = np.zeros(24)
    
    for i in range(items.shape[0]):
        if flag[int(items[i,6])] == 0:
            plotter.add_mesh(pv.Cube((0, 0, 0),0,0,0,(items[i,0],items[i,3],items[i,1],items[i,4],items[i,2],items[i,5])),opacity=1,color=colors[int(items[i,6])],style='wireframe',line_width=2,label=shapetypes[int(items[i,6])])
            flag[int(items[i,6])] = 1
        else:
            plotter.add_mesh(pv.Cube((0, 0, 0),0,0,0,(items[i,0],items[i,3],items[i,1],items[i,4],items[i,2],items[i,5])),opacity=1,color=colors[int(items[i,6])],style='wireframe',line_width=2)
                
    plotter.add_legend()
    plotter.show()
    