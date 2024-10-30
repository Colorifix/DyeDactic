from src.convert_spectrum_to_colour import *
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from src.utils import generate_count_fps, generate_colours, DrawMol



# define how to handle a hovering event
def hover(event):
    global data
    # if the mouse is over the scatter points
    if line.contains(event)[0]:
        # find out the index within the array from the event
        # if two points are too close to each other then take the first point
        if len(line.contains(event)[1]["ind"]) > 1:
            ind = line.contains(event)[1]["ind"][0]
        else:
            ind, = line.contains(event)[1]["ind"]
        # get the figure size
        w, h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.)
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0]*ws, xybox[1]*hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy = (fps_embedded[ind, 0], fps_embedded[ind, 1])
        # set the image corresponding to that point
        im.set_data(DrawMol(data["smiles"].iloc[ind], data["name"].iloc[ind],))
    else:
        #if the mouse is not over a scatter point
        ab.set_visible(False)
    fig.canvas.draw_idle()

data = pd.read_csv("data/pigments.csv")

fps = np.array([np.array(l) for l in data["smiles"].apply(generate_count_fps)])
fps_embedded = TSNE(n_components = 2, perplexity = 30).fit_transform(fps)
colours = generate_colours(data["lambda_max"])


# create figure and plot scatter
fig = plt.figure()
ax = fig.add_subplot(111)
line = ax.scatter(fps_embedded[:,0], fps_embedded[:,1], c=colours, edgecolors='black', linewidth=3, ls="", marker="o", alpha=0.5)
ax.set_xlabel("TSNE component 1")
ax.set_ylabel("TSNE component 2")

# create the annotations box
im = OffsetImage(np.zeros((100, 100)), zoom=1)
xybox=(100., 100.)
ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
        boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
# add it to the axes and make it invisible
ax.add_artist(ab)
ab.set_visible(False)

# add callback for mouse moves
fig.canvas.mpl_connect('motion_notify_event', hover)
plt.show()
