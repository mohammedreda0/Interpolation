from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg


app = pg.mkQApp()
win = pg.GraphicsLayoutWidget()
win.resize(1000, 600)
win.show()

plot = win.addPlot(y=np.random.normal(size=100, scale=10))

v_bar = pg.InfiniteLine(movable=True, angle=90)
plot.addItem(v_bar)


def handle_sig_dragged(obj):
    assert obj is v_bar
    print(obj.value())


v_bar.sigDragged.connect(handle_sig_dragged)

app.exec_()