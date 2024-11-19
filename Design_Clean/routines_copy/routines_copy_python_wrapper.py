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
_shared_object_name = "routines_copy." + platform.machine() + ".so"
_this_directory = os.path.dirname(os.path.abspath(__file__))
_path_to_lib = os.path.join(_this_directory, _shared_object_name)
_compile_options = ['-fPIC', '-shared', '-O3']
_ordered_dependencies = ['routines_copy.f90', 'routines_copy_c_wrapper.f90']
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


# ----------------------------------------------
# Wrapper for the Fortran subroutine EDDY_RANGE

def eddy_range(pos_vectors, factor):
    ''''''
    
    # Setting up "pos_vectors"
    if ((not issubclass(type(pos_vectors), numpy.ndarray)) or
        (not numpy.asarray(pos_vectors).flags.f_contiguous) or
        (not (pos_vectors.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'pos_vectors' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        pos_vectors = numpy.asarray(pos_vectors, dtype=ctypes.c_double, order='F')
    pos_vectors_dim_1 = ctypes.c_long(pos_vectors.shape[0])
    pos_vectors_dim_2 = ctypes.c_long(pos_vectors.shape[1])
    
    # Setting up "first_index"
    first_index = ctypes.c_int()
    
    # Setting up "last_index"
    last_index = ctypes.c_int()
    
    # Setting up "factor"
    if ((not issubclass(type(factor), numpy.ndarray)) or
        (not numpy.asarray(factor).flags.f_contiguous) or
        (not (factor.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'factor' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        factor = numpy.asarray(factor, dtype=ctypes.c_double, order='F')
    factor_dim_1 = ctypes.c_long(factor.shape[0])

    # Call C-accessible Fortran wrapper.
    clib.c_eddy_range(ctypes.byref(pos_vectors_dim_1), ctypes.byref(pos_vectors_dim_2), ctypes.c_void_p(pos_vectors.ctypes.data), ctypes.byref(first_index), ctypes.byref(last_index), ctypes.byref(factor_dim_1), ctypes.c_void_p(factor.ctypes.data))

    # Return final results, 'INTENT(OUT)' arguments only.
    return first_index.value, last_index.value, factor


# ----------------------------------------------
# Wrapper for the Fortran subroutine VELOCITY_GENERATOR

def velocity_generator(xaxis_trimmed, factor, u, v, w):
    ''''''
    
    # Setting up "xaxis_trimmed"
    if ((not issubclass(type(xaxis_trimmed), numpy.ndarray)) or
        (not numpy.asarray(xaxis_trimmed).flags.f_contiguous) or
        (not (xaxis_trimmed.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'xaxis_trimmed' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        xaxis_trimmed = numpy.asarray(xaxis_trimmed, dtype=ctypes.c_double, order='F')
    xaxis_trimmed_dim_1 = ctypes.c_long(xaxis_trimmed.shape[0])
    xaxis_trimmed_dim_2 = ctypes.c_long(xaxis_trimmed.shape[1])
    
    # Setting up "factor"
    if ((not issubclass(type(factor), numpy.ndarray)) or
        (not numpy.asarray(factor).flags.f_contiguous) or
        (not (factor.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'factor' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        factor = numpy.asarray(factor, dtype=ctypes.c_double, order='F')
    factor_dim_1 = ctypes.c_long(factor.shape[0])
    
    # Setting up "u"
    if ((not issubclass(type(u), numpy.ndarray)) or
        (not numpy.asarray(u).flags.f_contiguous) or
        (not (u.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'u' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        u = numpy.asarray(u, dtype=ctypes.c_double, order='F')
    u_dim_1 = ctypes.c_long(u.shape[0])
    
    # Setting up "v"
    if ((not issubclass(type(v), numpy.ndarray)) or
        (not numpy.asarray(v).flags.f_contiguous) or
        (not (v.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'v' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        v = numpy.asarray(v, dtype=ctypes.c_double, order='F')
    v_dim_1 = ctypes.c_long(v.shape[0])
    
    # Setting up "w"
    if ((not issubclass(type(w), numpy.ndarray)) or
        (not numpy.asarray(w).flags.f_contiguous) or
        (not (w.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'w' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        w = numpy.asarray(w, dtype=ctypes.c_double, order='F')
    w_dim_1 = ctypes.c_long(w.shape[0])

    # Call C-accessible Fortran wrapper.
    clib.c_velocity_generator(ctypes.byref(xaxis_trimmed_dim_1), ctypes.byref(xaxis_trimmed_dim_2), ctypes.c_void_p(xaxis_trimmed.ctypes.data), ctypes.byref(factor_dim_1), ctypes.c_void_p(factor.ctypes.data), ctypes.byref(u_dim_1), ctypes.c_void_p(u.ctypes.data), ctypes.byref(v_dim_1), ctypes.c_void_p(v.ctypes.data), ctypes.byref(w_dim_1), ctypes.c_void_p(w.ctypes.data))

    # Return final results, 'INTENT(OUT)' arguments only.
    return u, v, w


# ----------------------------------------------
# Wrapper for the Fortran subroutine SENSOR_LINE_GENERATOR

def sensor_line_generator(x_boundary, nxf, pos_vector=None):
    ''''''
    
    # Setting up "x_boundary"
    if (type(x_boundary) is not ctypes.c_double): x_boundary = ctypes.c_double(x_boundary)
    
    # Setting up "nxf"
    if (type(nxf) is not ctypes.c_int): nxf = ctypes.c_int(nxf)
    
    # Setting up "pos_vector"
    if (pos_vector is None):
        pos_vector = numpy.zeros(shape=(3, nxf), dtype=ctypes.c_double, order='F')
    elif ((not issubclass(type(pos_vector), numpy.ndarray)) or
          (not numpy.asarray(pos_vector).flags.f_contiguous) or
          (not (pos_vector.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'pos_vector' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        pos_vector = numpy.asarray(pos_vector, dtype=ctypes.c_double, order='F')
    pos_vector_dim_1 = ctypes.c_long(pos_vector.shape[0])
    pos_vector_dim_2 = ctypes.c_long(pos_vector.shape[1])

    # Call C-accessible Fortran wrapper.
    clib.c_sensor_line_generator(ctypes.byref(x_boundary), ctypes.byref(nxf), ctypes.byref(pos_vector_dim_1), ctypes.byref(pos_vector_dim_2), ctypes.c_void_p(pos_vector.ctypes.data))

    # Return final results, 'INTENT(OUT)' arguments only.
    return pos_vector

