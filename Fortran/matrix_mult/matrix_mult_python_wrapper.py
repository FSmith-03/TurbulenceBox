'''This Python code is an automatically generated wrapper
for Fortran code made by 'fmodpy'. The original documentation
for the Fortran source code follows.

! This module provides various testbed routines for demonstrating
! the simplicity of Fortran code usage with `fmodpy`.
!
! Contains:
!
!   MATRIX_MULTIPLY  --  A routine for multiplying two matrices of floats.
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
_shared_object_name = "matrix_mult." + platform.machine() + ".so"
_this_directory = os.path.dirname(os.path.abspath(__file__))
_path_to_lib = os.path.join(_this_directory, _shared_object_name)
_compile_options = ['-fPIC', '-shared', '-O3']
_ordered_dependencies = ['matrix_mult.f90', 'matrix_mult_c_wrapper.f90']
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
# Wrapper for the Fortran subroutine MATRIX_MULTIPLY

def matrix_multiply(a, b, out=None):
    '''! This subroutine multiplies the matrices A and B.
!
! INPUT:
!   A(M,N)  --  A 2D matrix of 64 bit floats.
!   B(N,P)  --  A 2D matrix of 64 bit floats,
!
! OUTPUT:
!   OUT(M,P)  --  The matrix that is the result of (AB).'''
    
    # Setting up "a"
    if ((not issubclass(type(a), numpy.ndarray)) or
        (not numpy.asarray(a).flags.f_contiguous) or
        (not (a.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'a' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        a = numpy.asarray(a, dtype=ctypes.c_double, order='F')
    a_dim_1 = ctypes.c_long(a.shape[0])
    a_dim_2 = ctypes.c_long(a.shape[1])
    
    # Setting up "b"
    if ((not issubclass(type(b), numpy.ndarray)) or
        (not numpy.asarray(b).flags.f_contiguous) or
        (not (b.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'b' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        b = numpy.asarray(b, dtype=ctypes.c_double, order='F')
    b_dim_1 = ctypes.c_long(b.shape[0])
    b_dim_2 = ctypes.c_long(b.shape[1])
    
    # Setting up "out"
    if (out is None):
        out = numpy.zeros(shape=(a.shape[0], b.shape[1]), dtype=ctypes.c_double, order='F')
    elif ((not issubclass(type(out), numpy.ndarray)) or
          (not numpy.asarray(out).flags.f_contiguous) or
          (not (out.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'out' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        out = numpy.asarray(out, dtype=ctypes.c_double, order='F')
    out_dim_1 = ctypes.c_long(out.shape[0])
    out_dim_2 = ctypes.c_long(out.shape[1])

    # Call C-accessible Fortran wrapper.
    clib.c_matrix_multiply(ctypes.byref(a_dim_1), ctypes.byref(a_dim_2), ctypes.c_void_p(a.ctypes.data), ctypes.byref(b_dim_1), ctypes.byref(b_dim_2), ctypes.c_void_p(b.ctypes.data), ctypes.byref(out_dim_1), ctypes.byref(out_dim_2), ctypes.c_void_p(out.ctypes.data))

    # Return final results, 'INTENT(OUT)' arguments only.
    return out

