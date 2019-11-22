# -*- coding: utf-8 -*-
from string import Template
import re

with open("tmp/netlist_1.cir", "rt") as f:
    netlist = f.read()

test_table = (".func f_C_ISS(v_DG) {TABLE(v_DG, 0.0, 9622.0E-12, 6.6, "
              "6088.0E-12, 15.9, 5038.7E-12, 28.9, 4741.4E-12, 47.5, "
              "4499.5E-12, 85.0, 4401.8E-12ohms, 601.2, 4469.3pf)}")

    
class SpiceTemplate(Template):
    delimiter = "_$_"


class SpiceConversions():
    si_map = {None: "", "f": "E-15", "p": "E-12", "n": "E-9", "u": "E-6", 
              "m": "E-3", "k": "E3", "meg": "E6", "g": "E9", "t": "E12"}
    # matches valid spice numbers with optional scale suffix and unit
    _join_lines_strip_comments_re = re.compile(r"\n[ \t]*\+|[ \t]*\*.*")
    _suffix_re = re.compile(
            r"([({,=] *\d*\.?\d+)([fpnumkgt]|(?:meg))?(e[+-]\d+)?[a-z]*",
            re.IGNORECASE)
    _tuple_re = re.compile(r"((?:\d*\.?\d+)(?:[eE][+-]?\d+)?),[ \t]*"
                           r"((?:\d*\.?\d+)(?:[eE][+-]?\d+)?)")

    def normalize(self, netlist: str) -> str:
        """Normalize a Spice netlist
        
        This:
            - replaces the spice number format ("4.7kohms") by
              standard floating-point string notation,
              
            - removes also the units attached to the numbers,
            
            - removes all comments from the file and
            
            - concatenates lines joined by a line-continuation ("+")
              character.
        """
        stripped_netlist = self._join_lines_strip_comments_re.sub("", netlist)
        def join_number(m):
            return m[1] + self.si_map[m[2]] + ("" if m[3] is None else m[3])
        return self._suffix_re.sub(join_number, stripped_netlist)
    
    def table_to_tuples(self, spice_table: str):
        """Render the interleaved values from a spice piece-wise linear
        lookup table ("TABLE") as a list of tuples.
        """
        float_str = self.normalize(spice_table)
        matches_iterator = self._tuple_re.finditer(float_str)
        return [(float(m[1]), float(m[2])) for m in matches_iterator]

    def table_to_lists(self, spice_table: str):
        """Render the interleaved values from a spice piece-wise linear
        lookup table ("TABLE") as two lists of corresponding X and Y data.
        
        Output is a tuple of these two lists
        """
        tuples = self.table_to_tuples(spice_table)
        return ([i for i, j in tuples], [j for i, j in tuples])
    
    def lists_to_table(self, name: str, arg: str, pwl_lists: tuple):
        tuples = [f"{arg:g}, {val:g}" for arg, val in zip(*pwl_lists)]
        tuples_str = ", ".join(tuples)
        return f".func {name}({arg}) {{TABLE({arg}, {tuples_str})}}"
    

sc = SpiceConversions()