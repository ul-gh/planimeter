#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export functions for Plot Model Assistant multi-ölot physical models
"""
class DefaultExporter():
    def __init__(self, phys_model):
        self.pma = phys_model.pma
        self.plots = phys_model.plots

    def export_as(filename: str):
        print("Not yet implemented, export as: ", filename)


class LTSpice_MOSFET(DefaultExporter):
    def __init__(self, phys_model):
        super().__init__(phys_model)


class Redmont_XLSX(DefaultExporter):
    def __init__(self, phys_model):
        super().__init__(phys_model)