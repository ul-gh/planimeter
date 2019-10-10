#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Some Customised Standard Widgets

License: GPL version 3
2019-10-10 Ulrich Lukas
"""
from functools import partial
from numpy import NaN, isnan

from PyQt5.QtCore import Qt, QLocale, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
        QHBoxLayout, QWidget, QLineEdit, QPushButton, QCheckBox, QMessageBox)


class StyledButton(QPushButton):
    """This checkable button has a minimum size set to the initial
    text contents requirements and can be used as a state indicator by
    connecting the set_green slot.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCheckable(True)
        size_ref_text = "A" + self.text() + "A"
        min_size = self.fontMetrics().size(Qt.TextShowMnemonic, size_ref_text)
        self.setMinimumSize(min_size)

    @pyqtSlot(bool)
    def set_green(self, state=True):
        style = "background-color: Palegreen" if state else ""
        self.setStyleSheet(style)


class NumberedButton(StyledButton):
    """This subclass of QPushButton adds a number index property and
    emits a corresponding signal emitting index and new state.
    """
    i_toggled = pyqtSignal(int, bool)
    def __init__(self, index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index = index
        self.toggled.connect(partial(self.i_toggled.emit, index))

    def index(self) ->int:
        return self._index

    @pyqtSlot(int)
    def setIndex(self, index: int):
        self._index = index


class SciLineEdit(QLineEdit):
    """QLineEdit with added validator for scientific notation input.
    Also, this takes a number as preset value instead of a string
    and holds a validated number property.
    """
    valid_number_entered = pyqtSignal(float)

    def __init__(
            self,
            preset_value=NaN,
            placeholderText="",
            num_fmt="G",
            *args,
            **kwargs
            ):
        text = "" if isnan(preset_value) else f"{preset_value:{num_fmt}}"
        super().__init__(
                text,
                *args,
                placeholderText=placeholderText,
                **kwargs
                )
        self.editingFinished.connect(self._update_value)
        # Defaults to notation=QDoubleValidator.ScientificNotation
        validator = QDoubleValidator(self)
        validator.setLocale(QLocale("en_US"))
        self.setValidator(validator)
        self._num_fmt = num_fmt
        self._value = preset_value
        self.setStyleSheet(":read-only {background-color: lightGrey;}")
    
    def value(self) -> float:
        return self._value
    @pyqtSlot(float)
    def setValue(self, value: float):
        self._value = value
        text = "" if isnan(value) else f"{value:{self._num_fmt}}"
        self.setText(text)

    @pyqtSlot()
    def _update_value(self):
        self._value = float(self.text().replace(",", "."))
        self.valid_number_entered.emit(self._value)


class SmallSciLineEdit(SciLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sizeHint(self):
        fm = self.fontMetrics()
        return fm.size(0, f"{8.888888888e+300:{self._num_fmt}}")

    def minimumSizeHint(self):
        return self.sizeHint()


class NumberedCenteredCheckbox(QWidget):
    # This is a checkbox inside a QWidget to get centered alignment - see:
    # https://bugreports.qt.io/browse/QTBUG-5368
    i_toggled = pyqtSignal(int, bool)

    def __init__(self, index, parent, *args, **kwargs):
        super().__init__(parent)
        self._index = index
        self.cbox = QCheckBox(self, *args, **kwargs)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.cbox)
        self.cbox.toggled.connect(partial(self.i_toggled.emit, index))

    def isChecked(self):
        return self.cbox.isChecked()
    
    @pyqtSlot(bool)
    def setChecked(self, state):
        self.cbox.setChecked(state)

    def index(self) ->int:
        return self._index

    @pyqtSlot(int)
    def setIndex(self, index: int):
        self._index = index


class CustomisedMessageBox(QMessageBox):
    """Messagebox with some common standard modal methods
    """
    def __init__(self, parent):
        super().__init__(parent)
        
    @pyqtSlot(Exception)
    def show_error(self, e: Exception):
        self.setIcon(QMessageBox.Critical)
        self.setText("<b>Error!</b>")
        text = f"Uncaught exception of {type(e)} occurred:\n{str(e)}"
        if hasattr(e, "filename"):
            text += f"\nFilename: {e.filename}"
        self.setInformativeText(text)
        self.setWindowTitle("Plot Workbench Error")
        return self.exec_()

    @pyqtSlot(str)
    def show_warning(self, text: str):
        self.setIcon(QMessageBox.Warning)
        self.setText("<b>Please note:</b>")
        self.setInformativeText(text)
        self.setWindowTitle("Plot Workbench Notification")
        return self.exec_()

    @pyqtSlot(str)
    def confirm_delete(self, message: str) -> bool:
        self.setIcon(QMessageBox.Warning)
        self.setText("<b>Deleting Items!</b>")
        self.setInformativeText(message)
        self.setWindowTitle("Confirm Delete")
        self.setStandardButtons(QMessageBox.Save | QMessageBox.Discard)
        return self.exec_() == QMessageBox.Discard
