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
    # This must be sorted by placing longest key strings first, i.e. "meg"
    si_map = {"f": "E-15", "p": "E-12", "n": "E-9", "u": "E-6", 
              "m": "E-3", "k": "E3", "meg": "E6", "g": "E9", "t": "E12"}
    _sorted_keys = sorted(si_map)
    _replacer_re = re.compile("|".join(_sorted_keys), re.IGNORECASE)
    _replacer = lambda self, m: self._replacer_re.sub(
                        lambda n: self.si_map[n[0].lower()], m[0])
    _finder_re = re.compile("(\d*\.?\d+(?:[fpumkg]?|(?:meg)?))", re.IGNORECASE)

    def spice_to_float_str(self, s: str) -> str:
        return self._finder_re.sub(self._replacer, s)