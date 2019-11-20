# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:17:26 2019

@author: lukas01
"""
#from multireplace import multireplace
from string import Template
import re

with open("tmp/netlist_1.cir", "rt") as f:
    netlist = f.read()

replacements = {"$r_1": "Foo_123"}
replacements_wo = {"r_1": "Foo_123"}

class SpiceTemplate(Template):
    delimiter = "_$_"



class SpiceConversions():
    si_map = {"a": "E-18", "f": "E-15", "p": "E-12", "n": "E-9", "u": "E-6", 
              "m": "E-3", "k": "E3", "meg": "E6", "g": "E9", "t": "E12"}
    _suffix_re = re.compile(r"(\d*\.?\d+)([afpnumkgt]|(?:meg))", re.IGNORECASE)
    _tuple_re = re.compile(r"((?:\d*\.?\d+)(?:[eE][+-]?\d+)?),[ \t]*"
                           r"((?:\d*\.?\d+)(?:[eE][+-]?\d+)?)")

    def spice_to_float_str(self, spice_number_str: str) -> str:
        return self._suffix_re.sub(lambda m: m[1] + self.si_map[m[2]],
                                   spice_number_str)
    
    def spice_to_list(self, spice_number_str: str):
        float_str = self.spice_to_float_str(spice_number_str)
        matches_iterator = self._tuple_re.finditer(float_str)
        return [[float(m[1]), float(m[2])] for m in matches_iterator]

sc = SpiceConversions()