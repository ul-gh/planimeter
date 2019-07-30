# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:29:25 2019
"""
from PyQt5.QtGui import QKeySequence
import PyQt5.QtWidgets as gui
import IPython

# When running in ipython, set up Qt integration
try:
    from IPython import get_ipython
    ipy_inst = get_ipython()
    if ipy_inst is not None:
        ipy_inst.run_line_magic("matplotlib", "qt5")
        ipy_inst.run_line_magic("gui", "qt")
except ImportError:
    ipy_inst = None



class Widget(gui.QWidget):
    def __init__(self,parent=None):
        gui.QWidget.__init__(self,parent)
        # initially construct the visible table
        self.tv=gui.QTableWidget()
        self.tv.setRowCount(1)
        self.tv.setColumnCount(1)
        self.tv.show()

        # set the shortcut ctrl+v for paste
        gui.QShortcut(QKeySequence('Ctrl+v'),self).activated.connect(self._handlePaste)

        self.layout = gui.QVBoxLayout(self)
        self.layout.addWidget(self.tv)



    # paste the value  
    def _handlePaste(self):
        clipboard_text = gui.QApplication.instance().clipboard().text()
        item.setText(clipboard_text)
        self.tv.setItem(0, 0, item)
        print(clipboard_text)
        print("möö")


app = gui.QApplication([])

w = Widget()
w.show()

cb = app.clipboard()
cbm = cb.mimeData()
formats = cbm.formats()

print(formats)
