# -*- coding: utf-8 -*-
import cx_Freeze
import sys
import os

base = None
os.environ["TCL_LIBRARY"] = r"D:\software\anaconda3\tcl\tcl8.6"
os.environ["TK_LIBRARY"] = r"D:\software\anaconda3\tcl\tk8.6"

# GUI applications require a different base on Windows (the default is for a
# console application).
if sys.platform == "win32":
    base = "Win32GUI"
#if sys.platform == "win32":
#    base = "console"

executables = [cx_Freeze.Executable("main.py", base = base)] #, icon="Icon.ico")]

cx_Freeze.setup(
    name="pwb",
    options = {"build_exe": {"packages":["tkinter"], "include_files":[r"D:\software\anaconda3\DLLs\tcl86t.dll",
                                                                      r"D:\software\anaconda3\DLLs\tk86t.dll"]}
        },  
    version = "1.0",
    description = "main",
    executables = executables
    )