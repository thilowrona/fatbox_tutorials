# -*- coding: utf-8 -*-
"""
created on Wed March 1 12:58:13 2023

@author: KELREY
"""

import fatbox
import numpy as np 
import networkx as nx
import pickle
import cv2
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from scipy.spatial import distance_matrix
from skimage import feature, morphology, filters
from sklearn.preprocessing import normalize

from sys import stdout

from fatbox.preprocessing import *
from fatbox.metrics import *
from fatbox.edits import *
from fatbox.plots import *

from networkx.readwrite import json_graph;

from mpl_toolkits.mplot3d import Axes3D
import plotly.offline as pyoff
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import chart_studio.plotly as py 

# for inlines
def processing_full_inline(data, plot=False, get_3d_data=False):
    # 1. open the data
    img = Image.open(data).convert('L')
    seismic_image = np.array(img)
    ## normalize the data
    normalize_seismic_image = 1-(seismic_image-np.min(seismic_image))/(np.max(seismic_image)-np.min(seismic_image))
    ## add gaussian blur/gaussian noise
    smoothed_seismic_image = filters.gaussian(normalize_seismic_image, sigma=1.2)
    ## binarized the data
    threshold = simple_threshold_binary(smoothed_seismic_image, 0.77)
    ## skeletonizeing the binarized data
    skeleton = skeleton_zhang_suen(threshold)
    ## remocing small faults
    removed_small = remove_small_regions(skeleton, 12)

    ## connecting the components
    ## ----------- if else condition ---------------------- ##
    ### connectedComponents
    ret, markers = cv2.connectedComponents(skeleton, connectivity=8)
    ### connectedComponents
    markers_copy = np.uint8(markers)
    markers_CWSA = cv2.connectedComponentsWithStatsWithAlgorithm(markers_copy, connectivity=8, ltype = cv2.CV_16U, ccltype=cv2.CCL_DEFAULT)
    ret_WSA, markers_CWSA_plot = markers_CWSA[0], markers_CWSA[1]
    ## ----------- if else condition ---------------------- ##

    ## make empty nx.Graph, defines point, add the 'pos' and 'components'
    G = nx.Graph()
    node = 0
    for comp in tqdm(range(1,ret)):

        points = np.transpose(np.vstack((np.where(markers==comp))))    
        
        for point in points:
            G.add_node(node)
            G.nodes[node]['pos'] = (point[1], point[0])
            G.nodes[node]['component'] = comp
            node += 1  
    ## add edge
    for comp in tqdm(range(1,ret)): 
    
        points = [G.nodes[node]['pos'] for node in G if G.nodes[node]['component']==comp]
        nodes  = [node for node in G if G.nodes[node]['component']==comp]

        dm = distance_matrix(points, points)  
        
        for n in range(len(points)):
            # print(n)
            for m in range(len(points)):
                if dm[n,m]<2.5 and n != m:
                    G.add_edge(nodes[n],nodes[m])
    ## split triple juction & Labeling components, removing(iterative) 
    G_labeled = label_components(G)
    G_labeled = simplify(G_labeled,2)
    
    G_split_triple = split_triple_junctions(G_labeled, dos=1, split='minimum' , threshold=2)
    G_split_triple = remove_small_components(G_split_triple, 24)
    G_split_triple = label_components(G_split_triple)
    ## Compute edge length
    G_edge_len = compute_edge_length(G_split_triple)
    ## calculate strike
    G_calculate_strike = calculate_strike(G_edge_len, 3)
    ## comp_to_fault
    G_component_to_fault = comp_to_fault(G_calculate_strike)

    ## Plotting
    if plot:
        # plot_labels = input('Do you wanna plot the labels? y/n')
        # if plot_labels == 'y':
        #     label = True
        # else:
        #     label = False

        fig, ax = plt.subplots(1, 1, figsize=(15,15))
        ax.imshow(np.zeros_like(seismic_image), 'gray_r', vmin=0)
        plt.title(f'Inline {data[88:-4]} - Label of Predicted Fault')
        plot_components(G_component_to_fault, label=True, node_size=1, ax=ax)
        # plot_faults(G, label=True, node_size=1, ax=ax)
        ax.set_xlim(0,seismic_image.shape[1])
        ax.set_ylim(seismic_image.shape[0],0)
        plt.show()

    if get_3d_data:
        inline = (data[88:-4])
        G_copy = G_component_to_fault

        node = 0
        for comp in tqdm(range(1,ret)):

            points = np.transpose(np.vstack((np.where(markers==comp))))    
            
            for point in points:
                G_copy.add_node(node)
                G_copy.nodes[node]['pos'] = (int(inline), point[1], point[0])
                G_copy.nodes[node]['component'] = comp

                node += 1  

        json_G_3d = json_graph.node_link_data(G_copy)

    ## Plot rose diagram
    # plot_rose(G_component_to_fault)
    
    
    if get_3d_data:
        print('Inline: ', data[88:-4])
        print('return 2 data, original & data with xyz coordinates')
        return G_component_to_fault, G_copy
    else:
        print('Inline: ', data[88:-4])
        return G_component_to_fault


# for xlines
def processing_full_xline(data, plot=False, get_3d_data=False):

    # 1. open the data
    img = Image.open(data).convert('L')
    seismic_image = np.array(img)
    ## normalize the data
    normalize_seismic_image = 1-(seismic_image-np.min(seismic_image))/(np.max(seismic_image)-np.min(seismic_image))
    ## add gaussian blur/gaussian noise
    smoothed_seismic_image = filters.gaussian(normalize_seismic_image, sigma=1.2)
    ## binarized the data
    threshold = simple_threshold_binary(smoothed_seismic_image, 0.77)
    ## skeletonizeing the binarized data
    skeleton = skeleton_zhang_suen(threshold)
    ## remocing small faults
    removed_small = remove_small_regions(skeleton, 12)

    ## connecting the components
    ## ----------- if else condition ---------------------- ##
    ### connectedComponents
    ret, markers = cv2.connectedComponents(skeleton, connectivity=8)
    ### connectedComponents
    markers_copy = np.uint8(markers)
    markers_CWSA = cv2.connectedComponentsWithStatsWithAlgorithm(markers_copy, connectivity=8, ltype = cv2.CV_16U, ccltype=cv2.CCL_DEFAULT)
    ret_WSA, markers_CWSA_plot = markers_CWSA[0], markers_CWSA[1]
    ## ----------- if else condition ---------------------- ##

    ## make empty nx.Graph, defines point, add the 'pos' and 'components'
    G = nx.Graph()
    node = 0
    for comp in tqdm(range(1,ret)):

        points = np.transpose(np.vstack((np.where(markers==comp))))    
        
        for point in points:
            G.add_node(node)
            G.nodes[node]['pos'] = (point[1], point[0])
            G.nodes[node]['component'] = comp
            node += 1  
    ## add edge
    for comp in tqdm(range(1,ret)): 
    
        points = [G.nodes[node]['pos'] for node in G if G.nodes[node]['component']==comp]
        nodes  = [node for node in G if G.nodes[node]['component']==comp]

        dm = distance_matrix(points, points)  
        
        for n in range(len(points)):
            # print(n)
            for m in range(len(points)):
                if dm[n,m]<2.5 and n != m:
                    G.add_edge(nodes[n],nodes[m])
    ## split triple juction & Labeling components, removing(iterative) 
    G_labeled = label_components(G)
    G_labeled = simplify(G_labeled,2)
    
    G_split_triple = split_triple_junctions(G_labeled, dos=1, split='minimum' , threshold=2)
    G_split_triple = remove_small_components(G_split_triple, 24)
    G_split_triple = label_components(G_split_triple)
    ## Compute edge length
    G_edge_len = compute_edge_length(G_split_triple)
    ## calculate strike
    G_calculate_strike = calculate_strike(G_edge_len, 3)
    ## comp_to_fault
    G_component_to_fault = comp_to_fault(G_calculate_strike)

    ## Plotting
    if plot:
        # plot_labels = input('Do you wanna plot the labels? y/n')
        # if plot_labels == 'y':
        #     label = True
        # else:
        #     label = False

        fig, ax = plt.subplots(1, 1, figsize=(15,15))
        ax.imshow(np.zeros_like(seismic_image), 'gray_r', vmin=0)
        plt.title(f'Xline {data[67:-4]} - Label of Predicted Fault')
        plot_components(G_component_to_fault, label=True, node_size=1, ax=ax)
        # plot_faults(G, label=True, node_size=1, ax=ax)
        ax.set_xlim(0,seismic_image.shape[1])
        ax.set_ylim(seismic_image.shape[0],0)
        plt.show()

    if get_3d_data:
        xline = (data[67:-4])
        G_copy = G_component_to_fault

        node = 0
        for comp in tqdm(range(1,ret)):

            points = np.transpose(np.vstack((np.where(markers==comp))))    
            
            for point in points:
                G_copy.add_node(node)
                G_copy.nodes[node]['pos'] = (point[1], int(xline), point[0])
                G_copy.nodes[node]['component'] = comp

                node += 1  

        json_G_3d = json_graph.node_link_data(G_copy)

    ## Plot rose diagram
    # plot_rose(G_component_to_fault)
    
    
    if get_3d_data:
        print('Xline: ', data[67:-4])
        print('return 2 data, original & data with xyz coordinates')
        return G_component_to_fault, G_copy
    else:
        print('Xline: ', data[67:-4])
        return G_component_to_fault

# for tlines
def processing_full_tline(data, plot=False, get_3d_data=False):

    # 1. open the data
    img = Image.open(data).convert('L')
    seismic_image = np.array(img)
    ## normalize the data
    normalize_seismic_image = 1-(seismic_image-np.min(seismic_image))/(np.max(seismic_image)-np.min(seismic_image))
    ## add gaussian blur/gaussian noise
    smoothed_seismic_image = filters.gaussian(normalize_seismic_image, sigma=1.2)
    ## binarized the data
    threshold = simple_threshold_binary(smoothed_seismic_image, 0.77)
    ## skeletonizeing the binarized data
    skeleton = skeleton_zhang_suen(threshold)
    ## remocing small faults
    removed_small = remove_small_regions(skeleton, 12)

    ## connecting the components
    ## ----------- if else condition ---------------------- ##
    ### connectedComponents
    ret, markers = cv2.connectedComponents(skeleton, connectivity=8)
    ### connectedComponents
    markers_copy = np.uint8(markers)
    markers_CWSA = cv2.connectedComponentsWithStatsWithAlgorithm(markers_copy, connectivity=8, ltype = cv2.CV_16U, ccltype=cv2.CCL_DEFAULT)
    ret_WSA, markers_CWSA_plot = markers_CWSA[0], markers_CWSA[1]
    ## ----------- if else condition ---------------------- ##

    ## make empty nx.Graph, defines point, add the 'pos' and 'components'
    G = nx.Graph()
    node = 0
    for comp in tqdm(range(1,ret)):

        points = np.transpose(np.vstack((np.where(markers==comp))))    
        
        for point in points:
            G.add_node(node)
            G.nodes[node]['pos'] = (point[1], point[0])
            G.nodes[node]['component'] = comp
            node += 1  
    ## add edge
    for comp in tqdm(range(1,ret)): 
    
        points = [G.nodes[node]['pos'] for node in G if G.nodes[node]['component']==comp]
        nodes  = [node for node in G if G.nodes[node]['component']==comp]

        dm = distance_matrix(points, points)  
        
        for n in range(len(points)):
            # print(n)
            for m in range(len(points)):
                if dm[n,m]<2.5 and n != m:
                    G.add_edge(nodes[n],nodes[m])
    ## split triple juction & Labeling components, removing(iterative) 
    G_labeled = label_components(G)
    G_labeled = simplify(G_labeled,2)
    
    G_split_triple = split_triple_junctions(G_labeled, dos=1, split='minimum' , threshold=2)
    G_split_triple = remove_small_components(G_split_triple, 24)
    G_split_triple = label_components(G_split_triple)
    ## Compute edge length
    G_edge_len = compute_edge_length(G_split_triple)
    ## calculate strike
    G_calculate_strike = calculate_strike(G_edge_len, 3)
    ## comp_to_fault
    G_component_to_fault = comp_to_fault(G_calculate_strike)

    ## Plotting
    if plot:
        # plot_labels = input('Do you wanna plot the labels? y/n')
        # if plot_labels == 'y':
        #     label = True
        # else:
        #     label = False

        fig, ax = plt.subplots(1, 1, figsize=(15,15))
        ax.imshow(np.zeros_like(seismic_image), 'gray_r', vmin=0)
        plt.title(f'Xline {data[67:-4]} - Label of Predicted Fault')
        plot_components(G_component_to_fault, label=True, node_size=1, ax=ax)
        # plot_faults(G, label=True, node_size=1, ax=ax)
        ax.set_xlim(0,seismic_image.shape[1])
        ax.set_ylim(seismic_image.shape[0],0)
        plt.show()

    if get_3d_data:
        tline = (data[67:-4])
        G_copy = G_component_to_fault

        node = 0
        for comp in tqdm(range(1,ret)):

            points = np.transpose(np.vstack((np.where(markers==comp))))    
            
            for point in points:
                G_copy.add_node(node)
                G_copy.nodes[node]['pos'] = (point[0], point[1], int(tline))
                G_copy.nodes[node]['component'] = comp

                node += 1  

        json_G_3d = json_graph.node_link_data(G_copy)

    ## Plot rose diagram
    # plot_rose(G_component_to_fault)
    
    
    if get_3d_data:
        print('Tline: ', data[67:-4])
        print('return 2 data, original & data with xyz coordinates')
        return G_component_to_fault, G_copy
    else:
        print('Tline: ', data[67:-4])
        return G_component_to_fault

def plot_fault_3d_matplt(data):
    from mpl_toolkits.mplot3d import Axes3D
    from chart_studio import plotly
    import plotly.offline as pyoff
    import plotly.graph_objs as go
    import numpy as np
    import matplotlib.pyplot as plt

    pos = nx.get_node_attributes(data, 'pos')
    xi, yi, zi = [], [], []
    for key, value in pos.items():
        xi.append(value[0])
        yi.append(value[1])
        zi.append(value[2])
    xi, yi, zi = np.array(xi), np.array(yi), np.array(zi)

    fig = plt.figure(figsize=(12,12))
    ax = plt.axes(projection="3d")
    ax.scatter(xi, yi,-1*zi, marker='o', c='b', edgecolors='w', alpha=0.7)
    # Axes3D.plot(xs=xi, ys = yi, zs=-1*zi, c='black', alpha=0.5)
    plt.show()


def plot_faults_3D_plotly(dict_of_data, slicing=False):
    import plotly.graph_objects as go
    import chart_studio.plotly as py 

    names   = list(dict_of_data.keys())
    nx_data = list(dict_of_data.values())
    
    if slicing:
        x0 = int(input('input min inline (int only): '))
        x1 = int(input('input max inline (int only): '))
        
        y0 = int(input('input min xline (int only): '))
        y1 = int(input('input max xline (int only): '))
        
        z0 = int(input('input min TWT (int only): '))
        z1 = int(input('input max TWT (int only): '))

    ## Storing the x,y,z values too dictionary
    empty_dict = {}
    
    ## for inline
    
    
    for i in dict_of_data.keys():

        edges = list(dict_of_data[i].nodes(data=True))
        x, y, z = [], [], []
        for e in edges:
            x.append(e[1]['pos'][0])
            y.append(e[1]['pos'][1])
            z.append(e[1]['pos'][2])
        empty_dict[i] = [x, y, z]

    data_for_plot = {}
    
    for i in empty_dict.keys():
        if i[0] == ('I' or 'i'):
            m = dict(symbol='circle',size=1.5,color='blue')
        if i[0] == ('X' or 'x'):
            m = dict(symbol='circle',size=1.5,color='red')
        if i[0] == ('T' or 't'):
            m = dict(symbol='circle',size=1.5,color='green')
            
        trace_nodes = go.Scatter3d(
                        x=empty_dict[i][0], 
                        y=empty_dict[i][1], 
                        z=empty_dict[i][2],
                        mode='markers',
                        marker=m,
                        name = i,
                        opacity=0.05
                        )
        data_for_plot[i] = trace_nodes

    data = [data_for_plot[i] for i in data_for_plot.keys()]
    
    layout = go.Layout(title="Fault Plane (in nodes) Extraction",
                       xaxis=dict(title="In-Line"),
                       yaxis=dict(title="X-Line"))
    
    fig = go.Figure(layout=layout, data=data)
    fig.update_layout(scene = dict(
                          xaxis_title='Xline',
                          yaxis_title='Inline',
                          zaxis_title='TWT (ms)'),
                      width=700,
                      margin=dict(r=20, b=10, l=10, t=10))
    if slicing:
        fig.update_layout(scene = dict(
                              xaxis = dict(nticks=4, range=[x0,x1]),
                              yaxis = dict(nticks=4, range=[y0,y1]),
                              zaxis = dict(nticks=4, range=[z0,z1])),
                          width=700,
                          margin=dict(r=20, b=10, l=10, t=10))
    fig.show()
    return None
        
def Merge(dict1, dict2):
    res = dict1 | dict2
    return res