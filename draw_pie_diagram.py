import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from collections import Counter, OrderedDict
import numpy as np
from matplotlib.patches import ConnectionPatch

# A pie diagram for colourant classes

font = {'family' : 'normal',
        'weight': 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

data = pd.read_csv("data/pigments.csv")
classes = dict(Counter(data["class"]))
classes = {k.replace("_", " "): v for k, v in classes.items()}
minors = {}


for cl in classes.keys():
    if classes[cl] < 15:
        minors[cl] = classes[cl]

for cl in minors.keys():
    del classes[cl]


classes["others"] = sum([value for value in minors.values()])
classes = OrderedDict(sorted(classes.items(), key = lambda x: x[1], reverse =True))
minors = OrderedDict(sorted(minors.items(), key = lambda x: x[1], reverse =True))

# make figure and assign axis objects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15), gridspec_kw={'width_ratios': [3, 2]})
fig.subplots_adjust(wspace=0)


# pie chart parameters
labels = list(classes.keys())
overall_ratios = [classes[name]/ sum(classes.values()) for name in labels]
explode = [0.1] + [0.0] * (len(classes) - 1)


# rotate so that first wedge is split by the x-axis
angle = -180 * overall_ratios[0]
wedges, *_ = ax1.pie(overall_ratios,
                     autopct='%1.1f%%',
                     startangle=angle,
                     labels=labels,
                     explode=explode,
                     pctdistance = 0.9,
                     labeldistance = 1.2)

cnt_colour = 0
colours = ["blue", "orange", "green", "red",
           "darkcyan", "magenta", "brown", "yellow",
           "lightblue", "gold", "lime", "salmon",
           "cyan", "violet", "chocolate", "olive", "cornflowerblue"]

for pie_wedge in wedges:
    pie_wedge.set_edgecolor('white')
    pie_wedge.set_facecolor(colours[cnt_colour])
    cnt_colour += 1

# bar chart parameters
age_ratios = [l / sum(minors.values()) for l in minors.values()]
age_labels = list(minors.keys())
bottom = 1
width = .05

# Adding from the top matches the legend.
for j, (height, label) in enumerate(reversed([*zip(age_ratios, age_labels)])):
    bottom -= height
    bc = ax2.bar(.0, height, width,
                 bottom=bottom, color='C0',
                 label=label + f" - {height:.0%}",
                 alpha = 0.1 + 0.035 * j, edgecolor="black")
    #ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

ax2.set_title('Minor classes')
ax2.legend(loc='best', bbox_to_anchor=(0.6, 0.85))
ax2.axis('off')
ax2.set_xlim(- 2.5 * width, 2.5 * width)

# use ConnectionPatch to draw lines between the two plots
theta1, theta2 = wedges[0].theta1, wedges[0].theta2
center, r = wedges[0].center, wedges[0].r
bar_height = sum(age_ratios)

# draw top connecting line
x = r * np.cos(np.pi / 180 * theta2) + center[0]
y = r * np.sin(np.pi / 180 * theta2) + center[1]
con = ConnectionPatch(xyA=(- width / 2, bar_height), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
con.set_linewidth(3)
ax2.add_artist(con)

# draw bottom connecting line
x = r * np.cos(np.pi / 180 * theta1) + center[0]
y = r * np.sin(np.pi / 180 * theta1) + center[1]
con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
ax2.add_artist(con)
con.set_linewidth(3)

plt.show()
