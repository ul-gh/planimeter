# coding: utf-8
import numpy as np
import matplotlib.pyplot

fig, ax = matplotlib.pyplot.subplots(1)

ax.plot((1.0, 2.0, 3), (1.0, 2.0, 3), "ro", picker=10)
ax.plot(2.0, "go", picker=10)
ax.plot(3.0, "bo", picker=10)

fig.show()

picked_artist = None
ind = 0
last_picked = []
last_event = []

def move(e):
    if picked_artist is None:
        return
    xdata, ydata = picked_artist.get_data()
    xdata[ind] = e.xdata
    ydata[ind] = e.ydata
    picked_artist.set_data(xdata, ydata)
    fig.canvas.draw_idle()

def foopick(e):
    global picked_artist
    global ind
    if picked_artist is None:
        print("pick ARTIST", e.artist)
        picked_artist = e.artist
        ind, = e.ind
        print(picked_artist)
        last_picked.append(picked_artist)
        last_event.append(e)
        print(e)

def foorelease(e):
    print("Event type: ", type(e))
    global picked_artist
    if picked_artist is not None:
        print("Release artist: ", picked_artist)
        picked_artist = None

cid0 = fig.canvas.mpl_connect("pick_event", foopick)
cid1 = fig.canvas.mpl_connect("button_release_event", foorelease)
cid2 = fig.canvas.mpl_connect("motion_notify_event", move)
