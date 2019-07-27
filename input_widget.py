#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import partial

from numpy import isclose

from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox,
        QGroupBox, QLabel, QPushButton, QRadioButton, QCheckBox,
        )

class NumberedButton(QPushButton):
    """This subclass of QPushButton adds a number index property and
    emits a corresponding integer signal
    """
    i_clicked = pyqtSignal(int)
    def __init__(self, index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.clicked.connect(partial(self.i_clicked.emit, index))

class SciLineEdit(QLineEdit):
    """QLineEdit with added validator for scientific notation input.
    Also, this takes a number as preset value instead of a string.
    """
    valid_number_entered = pyqtSignal(float)
    def __init__(self, preset_value, placeholderText, *args, **kwargs):
        text = None if preset_value is None else f"{preset_value:G}"
        super().__init__(
            text, *args, placeholderText=placeholderText, **kwargs)
        v = QDoubleValidator(self, notation=QDoubleValidator.ScientificNotation)
        self.setValidator(v)
        self.editingFinished.connect(self._emit_number)

    @pyqtSlot()
    def _emit_number(self):
        number, is_valid = self.locale().toDouble(self.text())
        if is_valid:
            self.valid_number_entered.emit(number)


class InputWidget(QWidget):
    def __init__(self, parent, model):
        super().__init__(parent)
        # Shortcuts to the data model
        self.model = model
        self.x_ax = model.x_ax
        self.y_ax = model.y_ax

        # Qt widget setup
        self.setFixedHeight(70)
        # Error message box used in set_prop()
        self.messagebox = QMessageBox(self)
        
        # X Coordinate picker and input boxes
        self.btn_pick_x = QPushButton(
                "Pick X-axis\nCoords", self, checkable=True)
        self.btn_pick_x.setAutoExclusive(True)
        hbox = QHBoxLayout(self)
        self.xstartw = SciLineEdit(self.x_ax.pts_data[0], "X Axis Start Value")
        self.xendw = SciLineEdit(self.x_ax.pts_data[1], "X Axis End Value")
        hbox.addWidget(self.xstartw)
        hbox.addWidget(self.xendw)
        # Lin/log buttons
        self.btn_lin_x = QRadioButton("Lin")
        self.btn_log_x = QRadioButton("Log")
        self.btn_lin_x.setChecked(not self.x_ax.log_scale)
        self.btn_log_x.setChecked(self.x_ax.log_scale)
        hbox.addWidget(self.btn_lin_x)
        hbox.addWidget(self.btn_log_x)
        self.group_x = QGroupBox("Enter X Axis Start and End Values")
        self.group_x.setLayout(hbox)
        
        # Y Coordinate picker and input boxes
        self.btn_pick_y = QPushButton(
            "Pick Y-axis\nCoords", self, checkable=True)
        self.btn_pick_y.setAutoExclusive(True)
        hbox = QHBoxLayout(self)
        self.ystartw = SciLineEdit(self.y_ax.pts_data[0], "Y Axis Start Value")
        self.yendw = SciLineEdit(self.y_ax.pts_data[1], "Y Axis End Value")
        hbox.addWidget(self.ystartw)
        hbox.addWidget(self.yendw)
        # Lin/log buttons
        self.btn_lin_y = QRadioButton("Lin")
        self.btn_log_y = QRadioButton("Log")
        self.btn_log_y.setChecked(self.y_ax.log_scale)
        self.btn_lin_y.setChecked(not self.y_ax.log_scale)
        hbox.addWidget(self.btn_lin_y)
        hbox.addWidget(self.btn_log_y)
        self.group_y = QGroupBox("Enter Y Axis Start and End Values")
        self.group_y.setLayout(hbox)

        # Store plot config button
        self.btn_store_config = QCheckBox("Store Config")
        self.btn_store_config.setChecked(self.model.store_ax_conf)

        # Pick traces buttons
        self.btns_pick_trace = (
                NumberedButton(0, "Pick\nTrace 1", self, checkable=True),
                NumberedButton(1, "Pick\nTrace 2", self, checkable=True),
                NumberedButton(2, "Pick\nTrace 3", self, checkable=True),
                )
        
        # Dummy button acting as focus stealer for default mode
        self.btn_default = QPushButton("Drag-Drop\nMode", self)
        
        # This is all input boxes plus label
        hbox = QHBoxLayout(self)
        hbox.addWidget(self.btn_pick_x)
        hbox.addWidget(self.group_x)
        hbox.addWidget(self.btn_pick_y)
        hbox.addWidget(self.group_y)
        hbox.addWidget(self.btn_store_config)
        for i in self.btns_pick_trace:
            i.setAutoExclusive(True)
            #self.button_group.addButton(i)
            hbox.addWidget(i)
        hbox.addWidget(self.btn_default)
        self.setLayout(hbox)

        ########## Initialise view from model
        self.update_axes_view()


    ### Slots definition ###
    @pyqtSlot()
    def update_axes_view(self):
        """Updates buttons and input boxes to represent the data model state"""
        x_ax = self.x_ax
        y_ax = self.y_ax
        num_format = self.model.num_format
        # Update axis section value input boxes
        x0, x1, y0, y1 = *x_ax.pts_data, *y_ax.pts_data
        self.xstartw.setText("" if x0 is None else f"{x0:{num_format}}")
        self.xendw.setText("" if x1 is None else f"{x1:{num_format}}")
        self.ystartw.setText("" if y0 is None else f"{y0:{num_format}}")
        self.yendw.setText("" if y1 is None else f"{y1:{num_format}}")
        invalid_x = x_ax.log_scale and isclose(
            x_ax.pts_data, 0.0, atol=x_ax.atol).any()
        invalid_y = y_ax.log_scale and isclose(
            y_ax.pts_data, 0.0, atol=y_ax.atol).any()
        self.set_green_x_ax(not invalid_x and None not in x_ax.pts_data)
        self.set_green_y_ax(not invalid_y and None not in y_ax.pts_data)
        # Update log/lin radio buttons.
        self.btn_lin_x.setChecked(not x_ax.log_scale)
        self.btn_log_x.setChecked(x_ax.log_scale)
        self.btn_lin_y.setChecked(not y_ax.log_scale)
        self.btn_log_y.setChecked(y_ax.log_scale)
        # Store config button
        self.btn_store_config.setChecked(self.model.store_ax_conf)


    @pyqtSlot(bool)
    def set_green_x_ax(self, state):
        # Background set to green when model has valid data
        style = "QLineEdit { background-color: Palegreen; }" if state else ""
        self.group_x.setStyleSheet(style)

    @pyqtSlot(bool)
    def set_green_y_ax(self, state):
        # Background set to green when model has valid data
        style = "QLineEdit { background-color: Palegreen; }" if state else ""
        self.group_y.setStyleSheet(style)

    @pyqtSlot(bool)
    def set_green_btn_pick_x(self, state):
        style = "background-color: Palegreen" if state else ""
        self.btn_pick_x.setStyleSheet(style)

    @pyqtSlot(bool)
    def set_green_btn_pick_y(self, state):
        style = "background-color: Palegreen" if state else ""
        self.btn_pick_y.setStyleSheet(style)
    
    @pyqtSlot()
    def uncheck_all_buttons(self):
        self.btn_pick_x.setChecked(False)
        self.btn_pick_y.setChecked(False)
        for i in self.btns_pick_trace:
            i.setChecked(False)
        # Set focus to the default in order to unfocus all other buttons
        self.btn_default.setFocus()

