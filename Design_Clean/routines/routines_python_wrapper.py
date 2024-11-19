'''This Python code is an automatically generated wrapper
for Fortran code made by 'fmodpy'. The original documentation
for the Fortran source code follows.


'''

import os
import ctypes
import platform
import numpy

# --------------------------------------------------------------------
#               CONFIGURATION
# 
_verbose = True
_fort_compiler = "gfortran"
_shared_object_name = "routines." + platform.machine() + ".so"
_this_directory = os.path.dirname(os.path.abspath(__file__))
_path_to_lib = os.path.join(_this_directory, _shared_object_name)
_compile_options = ['-fPIC', '-shared', '-O3']
_ordered_dependencies = ['routines.f90', 'routines_c_wrapper.f90']
_symbol_files = []# 
# --------------------------------------------------------------------
#               AUTO-COMPILING
#
# Try to import the prerequisite symbols for the compiled code.
for _ in _symbol_files:
    _ = ctypes.CDLL(os.path.join(_this_directory, _), mode=ctypes.RTLD_GLOBAL)
# Try to import the existing object. If that fails, recompile and then try.
try:
    # Check to see if the source files have been modified and a recompilation is needed.
    if (max(max([0]+[os.path.getmtime(os.path.realpath(os.path.join(_this_directory,_))) for _ in _symbol_files]),
            max([0]+[os.path.getmtime(os.path.realpath(os.path.join(_this_directory,_))) for _ in _ordered_dependencies]))
        > os.path.getmtime(_path_to_lib)):
        print()
        print("WARNING: Recompiling because the modification time of a source file is newer than the library.", flush=True)
        print()
        if os.path.exists(_path_to_lib):
            os.remove(_path_to_lib)
        raise NotImplementedError(f"The newest library code has not been compiled.")
    # Import the library.
    clib = ctypes.CDLL(_path_to_lib)
except:
    # Remove the shared object if it exists, because it is faulty.
    if os.path.exists(_shared_object_name):
        os.remove(_shared_object_name)
    # Compile a new shared object.
    _command = [_fort_compiler] + _ordered_dependencies + _compile_options + ["-o", _shared_object_name]
    if _verbose:
        print("Running system command with arguments")
        print("  ", " ".join(_command))
    # Run the compilation command.
    import subprocess
    subprocess.check_call(_command, cwd=_this_directory)
    # Import the shared object file as a C library with ctypes.
    clib = ctypes.CDLL(_path_to_lib)
# --------------------------------------------------------------------


# ----------------------------------------------
# Wrapper for the Fortran subroutine MATRIX_ROTATE

def matrix_rotate(thetax, thetay, thetaz, r=None):
    ''''''
    
    # Setting up "thetax"
    if (type(thetax) is not ctypes.c_float): thetax = ctypes.c_float(thetax)
    
    # Setting up "thetay"
    if (type(thetay) is not ctypes.c_float): thetay = ctypes.c_float(thetay)
    
    # Setting up "thetaz"
    if (type(thetaz) is not ctypes.c_float): thetaz = ctypes.c_float(thetaz)
    
    # Setting up "r"
    if (r is None):
        r = numpy.zeros(shape=(3, 3), dtype=ctypes.c_float, order='F')
    elif ((not issubclass(type(r), numpy.ndarray)) or
          (not numpy.asarray(r).flags.f_contiguous) or
          (not (r.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'r' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        r = numpy.asarray(r, dtype=ctypes.c_float, order='F')
    r_dim_1 = ctypes.c_long(r.shape[0])
    r_dim_2 = ctypes.c_long(r.shape[1])

    # Call C-accessible Fortran wrapper.
    clib.c_matrix_rotate(ctypes.byref(thetax), ctypes.byref(thetay), ctypes.byref(thetaz), ctypes.byref(r_dim_1), ctypes.byref(r_dim_2), ctypes.c_void_p(r.ctypes.data))

    # Return final results, 'INTENT(OUT)' arguments only.
    return r

