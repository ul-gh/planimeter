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

test_table = (".func f_C_ISS(v_DG) {TABLE(v_DG, 0.0, 9622.0E-12, 6.6,"
              "6088.0E-12, 15.9, 5038.7E-12, 28.9, 4741.4E-12, 47.5, "
              "4499.5E-12, 85.0, 4401.8E-12, 601.2, 4469.3E-12)}")

replacements = {"$r_1": "Foo_123"}
replacements_wo = {"r_1": "Foo_123"}

class SpiceTemplate(Template):
    delimiter = "_$_"



class SpiceConversions():
    si_map = {None: "", "f": "E-15", "p": "E-12", "n": "E-9", "u": "E-6", 
              "m": "E-3", "k": "E3", "meg": "E6", "g": "E9", "t": "E12"}
    # matches valid spice numbers with optional scale suffix and unit
    _suffix_re = re.compile(r"[^a-z\d](\d*\.?\d+)([fpnumkgt]|(?:meg))?[a-z]*",
                            re.IGNORECASE)
    _tuple_re = re.compile(r"((?:\d*\.?\d+)(?:[eE][+-]?\d+)?),[ \t]*"
                           r"((?:\d*\.?\d+)(?:[eE][+-]?\d+)?)")

    def netlist_to_float(self, netlist: str) -> str:
        """Convert all numbers in a spice-compatible input string
        (e.g. a netlist) into valid floating point number strings.
        
        This removes all units from the number strings as well.
        """
        return self._suffix_re.sub(lambda m: m[1] + self.si_map[m[2]], netlist)
    
    def table_to_tuples(self, spice_table: str):
        """Render the interleaved values from a spice piece-wise linear
        lookup table ("TABLE") as a list of tuples.
        """
        float_str = self.netlist_to_float(spice_table)
        matches_iterator = self._tuple_re.finditer(float_str)
        return [(float(m[1]), float(m[2])) for m in matches_iterator]

    def table_to_lists(self, spice_table: str):
        """Render the interleaved values from a spice piece-wise linear
        lookup table ("TABLE") as two lists of corresponding X and Y data.
        
        Output is a tuple of these two lists
        """
        tuples = self.table_to_tuples(spice_table)
        return ([i for i, j in tuples], [j for i, j in tuples])

sc = SpiceConversions()