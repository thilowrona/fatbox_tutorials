# -*- coding: utf-8 -*-
"""
Created on Fri May  6 09:49:35 2022

@author: thilo
"""
import pickle
import networkx as nx
import matplotlib.pyplot as plt
plt.close("all")


from fatbox.preprocessing import *
from fatbox.edits import *
from fatbox.metrics import *
from fatbox.plots import *
from fatbox.utils import *




G = pickle.load(open('G.p', 'rb'))

mask = np.load('mask.npy')


# fig, ax = plt.subplots(1,1)
# ax.imshow(mask)
# plot_components(G, node_size=1, label=False, ax=ax)

d = 50



def pick_handler(event):
    if event.button==1 and event.xdata != None and event.ydata != None:
        
        x = int(event.xdata)
        y = int(event.ydata)
        
        mask[y-d:y+d, x-d:x+d] = 1
        
        plt.clf()
        plt.imshow(mask)
        ax = fig.gca()
        plot_components(G, node_size=1, label=False, ax=ax)
        ax.set_xlim(0, mask.shape[1])
        ax.set_ylim(mask.shape[0], 0)
        plt.draw() #redraw
        
        
        
    if event.button==3 and event.xdata != None and event.ydata != None:
        
        x = int(event.xdata)
        y = int(event.ydata)
        
        mask[y-d:y+d, x-d:x+d] = 0
        
        plt.clf()
        plt.imshow(mask)
        ax = fig.gca()
        plot_components(G, node_size=1, label=False, ax=ax)
        ax.set_xlim(0, mask.shape[1])
        ax.set_ylim(mask.shape[0], 0)
        plt.draw() #redraw        
        
        
        
        

fig, ax  =plt.subplots(figsize=(6,12))
ax.imshow(mask)
fig.canvas.mpl_connect('button_press_event', pick_handler)
# fig.canvas.mpl_connect('button_press_event', unpick_handler)

ax = fig.gca()
plot_components(G, node_size=1, label=False, ax=ax)
ax.set_xlim(0, mask.shape[1])
ax.set_ylim(mask.shape[0], 0)
plt.show()
plt.draw()


#%%
np.save('mask.npy', mask)


G = extract_attribute(G, mask, 'mask')

node_list = []
for comp in nx.connected_components(G):
    for node in comp:
        remove = False
        if G.nodes[node]['mask'] == 1:
            remove = True
        if remove:
            for node in comp:
                node_list.append(node)

G.remove_nodes_from(node_list)
            


fig, ax  =plt.subplots(figsize=(6,12))
ax.imshow(np.zeros_like(mask))
plot_components(G, node_size=1, label=False, ax=ax)
ax.set_xlim(0, mask.shape[1])
ax.set_ylim(mask.shape[0], 0)
plt.show()

pickle.dump(G, open('G_clean.p', "wb" ))