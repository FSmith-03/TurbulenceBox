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
_shared_object_name = "routines_simplified." + platform.machine() + ".so"
_this_directory = os.path.dirname(os.path.abspath(__file__))
_path_to_lib = os.path.join(_this_directory, _shared_object_name)
_compile_options = ['-fPIC', '-shared', '-O3']
_ordered_dependencies = ['routines_simplified.f90', 'routines_simplified_c_wrapper.f90']
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

def matrix_rotate(thetax, thetay, thetaz):
    ''''''
    
    # Setting up "thetax"
    if (type(thetax) is not ctypes.c_float): thetax = ctypes.c_float(thetax)
    
    # Setting up "thetay"
    if (type(thetay) is not ctypes.c_float): thetay = ctypes.c_float(thetay)
    
    # Setting up "thetaz"
    if (type(thetaz) is not ctypes.c_float): thetaz = ctypes.c_float(thetaz)
    
    # Setting up "r"
    r = ctypes.c_void_p()
    r_dim_1 = ctypes.c_long()
    r_dim_2 = ctypes.c_long()

    # Call C-accessible Fortran wrapper.
    clib.c_matrix_rotate(ctypes.byref(thetax), ctypes.byref(thetay), ctypes.byref(thetaz), ctypes.byref(r_dim_1), ctypes.byref(r_dim_2), ctypes.byref(r))

    # Post-processing "r"
    r_size = (r_dim_1.value) * (r_dim_2.value)
    if (r_size > 0):
        r = numpy.array(ctypes.cast(r, ctypes.POINTER(ctypes.c_float*r_size)).contents, copy=False)
        r = r.reshape(r_dim_2.value,r_dim_1.value).T
    elif (r_size == 0):
        r = numpy.zeros(shape=(r_dim_2.value,r_dim_1.value), dtype=ctypes.c_float, order='F')
    else:
        r = None
    
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
# Wrapper for the Fortran subroutine TRIMMER_INDEX

def trimmer_index(pos_vectors, first_and_last=None):
    ''''''
    
    # Setting up "pos_vectors"
    if ((not issubclass(type(pos_vectors), numpy.ndarray)) or
        (not numpy.asarray(pos_vectors).flags.f_contiguous) or
        (not (pos_vectors.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'pos_vectors' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        pos_vectors = numpy.asarray(pos_vectors, dtype=ctypes.c_float, order='F')
    pos_vectors_dim_1 = ctypes.c_long(pos_vectors.shape[0])
    pos_vectors_dim_2 = ctypes.c_long(pos_vectors.shape[1])
    
    # Setting up "first_and_last"
    if (first_and_last is None):
        first_and_last = numpy.zeros(shape=(2), dtype=ctypes.c_int, order='F')
    elif ((not issubclass(type(first_and_last), numpy.ndarray)) or
          (not numpy.asarray(first_and_last).flags.f_contiguous) or
          (not (first_and_last.dtype == numpy.dtype(ctypes.c_int)))):
        import warnings
        warnings.warn("The provided argument 'first_and_last' was not an f_contiguous NumPy array of type 'ctypes.c_int' (or equivalent). Automatically converting (probably creating a full copy).")
        first_and_last = numpy.asarray(first_and_last, dtype=ctypes.c_int, order='F')
    first_and_last_dim_1 = ctypes.c_long(first_and_last.shape[0])

    # Call C-accessible Fortran wrapper.
    clib.c_trimmer_index(ctypes.byref(pos_vectors_dim_1), ctypes.byref(pos_vectors_dim_2), ctypes.c_void_p(pos_vectors.ctypes.data), ctypes.byref(first_and_last_dim_1), ctypes.c_void_p(first_and_last.ctypes.data))

    # Return final results, 'INTENT(OUT)' arguments only.
    return first_and_last


# ----------------------------------------------
# Wrapper for the Fortran subroutine TRIMMER

def trimmer(first_and_last, xaxis):
    ''''''
    
    # Setting up "first_and_last"
    if ((not issubclass(type(first_and_last), numpy.ndarray)) or
        (not numpy.asarray(first_and_last).flags.f_contiguous) or
        (not (first_and_last.dtype == numpy.dtype(ctypes.c_int)))):
        import warnings
        warnings.warn("The provided argument 'first_and_last' was not an f_contiguous NumPy array of type 'ctypes.c_int' (or equivalent). Automatically converting (probably creating a full copy).")
        first_and_last = numpy.asarray(first_and_last, dtype=ctypes.c_int, order='F')
    first_and_last_dim_1 = ctypes.c_long(first_and_last.shape[0])
    
    # Setting up "xaxis"
    if ((not issubclass(type(xaxis), numpy.ndarray)) or
        (not numpy.asarray(xaxis).flags.f_contiguous) or
        (not (xaxis.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'xaxis' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        xaxis = numpy.asarray(xaxis, dtype=ctypes.c_float, order='F')
    xaxis_dim_1 = ctypes.c_long(xaxis.shape[0])
    xaxis_dim_2 = ctypes.c_long(xaxis.shape[1])
    
    # Setting up "xaxis_trimmed"
    xaxis_trimmed = ctypes.c_void_p()
    xaxis_trimmed_dim_1 = ctypes.c_long()
    xaxis_trimmed_dim_2 = ctypes.c_long()

    # Call C-accessible Fortran wrapper.
    clib.c_trimmer(ctypes.byref(first_and_last_dim_1), ctypes.c_void_p(first_and_last.ctypes.data), ctypes.byref(xaxis_dim_1), ctypes.byref(xaxis_dim_2), ctypes.byref(xaxis), ctypes.byref(xaxis_trimmed_dim_1), ctypes.byref(xaxis_trimmed_dim_2), ctypes.byref(xaxis_trimmed))

    # Post-processing "xaxis_trimmed"
    xaxis_trimmed_size = (xaxis_trimmed_dim_1.value) * (xaxis_trimmed_dim_2.value)
    if (xaxis_trimmed_size > 0):
        xaxis_trimmed = numpy.array(ctypes.cast(xaxis_trimmed, ctypes.POINTER(ctypes.c_float*xaxis_trimmed_size)).contents, copy=False)
        xaxis_trimmed = xaxis_trimmed.reshape(xaxis_trimmed_dim_2.value,xaxis_trimmed_dim_1.value).T
    elif (xaxis_trimmed_size == 0):
        xaxis_trimmed = numpy.zeros(shape=(xaxis_trimmed_dim_2.value,xaxis_trimmed_dim_1.value), dtype=ctypes.c_float, order='F')
    else:
        xaxis_trimmed = None
    
    # Return final results, 'INTENT(OUT)' arguments only.
    return xaxis_trimmed


# ----------------------------------------------
# Wrapper for the Fortran subroutine VELOCITY_CALC

def velocity_calc(xaxis_trimmed):
    ''''''
    
    # Setting up "xaxis_trimmed"
    if ((not issubclass(type(xaxis_trimmed), numpy.ndarray)) or
        (not numpy.asarray(xaxis_trimmed).flags.f_contiguous) or
        (not (xaxis_trimmed.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'xaxis_trimmed' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        xaxis_trimmed = numpy.asarray(xaxis_trimmed, dtype=ctypes.c_float, order='F')
    xaxis_trimmed_dim_1 = ctypes.c_long(xaxis_trimmed.shape[0])
    xaxis_trimmed_dim_2 = ctypes.c_long(xaxis_trimmed.shape[1])
    
    # Setting up "u"
    u = ctypes.c_void_p()
    u_dim_1 = ctypes.c_long()
    
    # Setting up "v"
    v = ctypes.c_void_p()
    v_dim_1 = ctypes.c_long()
    
    # Setting up "w"
    w = ctypes.c_void_p()
    w_dim_1 = ctypes.c_long()

    # Call C-accessible Fortran wrapper.
    clib.c_velocity_calc(ctypes.byref(xaxis_trimmed_dim_1), ctypes.byref(xaxis_trimmed_dim_2), ctypes.c_void_p(xaxis_trimmed.ctypes.data), ctypes.byref(u_dim_1), ctypes.byref(u), ctypes.byref(v_dim_1), ctypes.byref(v), ctypes.byref(w_dim_1), ctypes.byref(w))

    # Post-processing "u"
    u_size = (u_dim_1.value)
    if (u_size > 0):
        u = numpy.array(ctypes.cast(u, ctypes.POINTER(ctypes.c_float*u_size)).contents, copy=False)
    elif (u_size == 0):
        u = numpy.zeros(shape=(u_dim_1.value), dtype=ctypes.c_float, order='F')
    else:
        u = None
    
    # Post-processing "v"
    v_size = (v_dim_1.value)
    if (v_size > 0):
        v = numpy.array(ctypes.cast(v, ctypes.POINTER(ctypes.c_float*v_size)).contents, copy=False)
    elif (v_size == 0):
        v = numpy.zeros(shape=(v_dim_1.value), dtype=ctypes.c_float, order='F')
    else:
        v = None
    
    # Post-processing "w"
    w_size = (w_dim_1.value)
    if (w_size > 0):
        w = numpy.array(ctypes.cast(w, ctypes.POINTER(ctypes.c_float*w_size)).contents, copy=False)
    elif (w_size == 0):
        w = numpy.zeros(shape=(w_dim_1.value), dtype=ctypes.c_float, order='F')
    else:
        w = None
    
    # Return final results, 'INTENT(OUT)' arguments only.
    return u, v, w


# ----------------------------------------------
# Wrapper for the Fortran subroutine VECTOR_SUMS

def vector_sums(u, v, w, u_total, v_total, w_total, first_index, last_index):
    ''''''
    
    # Setting up "u"
    if ((not issubclass(type(u), numpy.ndarray)) or
        (not numpy.asarray(u).flags.f_contiguous) or
        (not (u.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'u' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        u = numpy.asarray(u, dtype=ctypes.c_float, order='F')
    u_dim_1 = ctypes.c_long(u.shape[0])
    
    # Setting up "v"
    if ((not issubclass(type(v), numpy.ndarray)) or
        (not numpy.asarray(v).flags.f_contiguous) or
        (not (v.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'v' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        v = numpy.asarray(v, dtype=ctypes.c_float, order='F')
    v_dim_1 = ctypes.c_long(v.shape[0])
    
    # Setting up "w"
    if ((not issubclass(type(w), numpy.ndarray)) or
        (not numpy.asarray(w).flags.f_contiguous) or
        (not (w.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'w' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        w = numpy.asarray(w, dtype=ctypes.c_float, order='F')
    w_dim_1 = ctypes.c_long(w.shape[0])
    
    # Setting up "u_total"
    if ((not issubclass(type(u_total), numpy.ndarray)) or
        (not numpy.asarray(u_total).flags.f_contiguous) or
        (not (u_total.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'u_total' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        u_total = numpy.asarray(u_total, dtype=ctypes.c_float, order='F')
    u_total_dim_1 = ctypes.c_long(u_total.shape[0])
    
    # Setting up "v_total"
    if ((not issubclass(type(v_total), numpy.ndarray)) or
        (not numpy.asarray(v_total).flags.f_contiguous) or
        (not (v_total.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'v_total' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        v_total = numpy.asarray(v_total, dtype=ctypes.c_float, order='F')
    v_total_dim_1 = ctypes.c_long(v_total.shape[0])
    
    # Setting up "w_total"
    if ((not issubclass(type(w_total), numpy.ndarray)) or
        (not numpy.asarray(w_total).flags.f_contiguous) or
        (not (w_total.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'w_total' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        w_total = numpy.asarray(w_total, dtype=ctypes.c_float, order='F')
    w_total_dim_1 = ctypes.c_long(w_total.shape[0])
    
    # Setting up "first_index"
    if (type(first_index) is not ctypes.c_int): first_index = ctypes.c_int(first_index)
    
    # Setting up "last_index"
    if (type(last_index) is not ctypes.c_int): last_index = ctypes.c_int(last_index)

    # Call C-accessible Fortran wrapper.
    clib.c_vector_sums(ctypes.byref(u_dim_1), ctypes.c_void_p(u.ctypes.data), ctypes.byref(v_dim_1), ctypes.c_void_p(v.ctypes.data), ctypes.byref(w_dim_1), ctypes.c_void_p(w.ctypes.data), ctypes.byref(u_total_dim_1), ctypes.c_void_p(u_total.ctypes.data), ctypes.byref(v_total_dim_1), ctypes.c_void_p(v_total.ctypes.data), ctypes.byref(w_total_dim_1), ctypes.c_void_p(w_total.ctypes.data), ctypes.byref(first_index), ctypes.byref(last_index))

    # Return final results, 'INTENT(OUT)' arguments only.
    return u_total, v_total, w_total


# ----------------------------------------------
# Wrapper for the Fortran subroutine MAIN_CALCULATION

def main_calculation(input_ints, a_list, theta_list):
    ''''''
    
    # Setting up "input_ints"
    if ((not issubclass(type(input_ints), numpy.ndarray)) or
        (not numpy.asarray(input_ints).flags.f_contiguous) or
        (not (input_ints.dtype == numpy.dtype(ctypes.c_int)))):
        import warnings
        warnings.warn("The provided argument 'input_ints' was not an f_contiguous NumPy array of type 'ctypes.c_int' (or equivalent). Automatically converting (probably creating a full copy).")
        input_ints = numpy.asarray(input_ints, dtype=ctypes.c_int, order='F')
    input_ints_dim_1 = ctypes.c_long(input_ints.shape[0])
    
    # Setting up "a_list"
    if ((not issubclass(type(a_list), numpy.ndarray)) or
        (not numpy.asarray(a_list).flags.f_contiguous) or
        (not (a_list.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'a_list' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        a_list = numpy.asarray(a_list, dtype=ctypes.c_float, order='F')
    a_list_dim_1 = ctypes.c_long(a_list.shape[0])
    a_list_dim_2 = ctypes.c_long(a_list.shape[1])
    
    # Setting up "theta_list"
    if ((not issubclass(type(theta_list), numpy.ndarray)) or
        (not numpy.asarray(theta_list).flags.f_contiguous) or
        (not (theta_list.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'theta_list' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        theta_list = numpy.asarray(theta_list, dtype=ctypes.c_float, order='F')
    theta_list_dim_1 = ctypes.c_long(theta_list.shape[0])
    theta_list_dim_2 = ctypes.c_long(theta_list.shape[1])
    
    # Setting up "velocity_total"
    velocity_total = ctypes.c_void_p()
    velocity_total_dim_1 = ctypes.c_long()
    velocity_total_dim_2 = ctypes.c_long()

    # Call C-accessible Fortran wrapper.
    clib.c_main_calculation(ctypes.byref(input_ints_dim_1), ctypes.c_void_p(input_ints.ctypes.data), ctypes.byref(a_list_dim_1), ctypes.byref(a_list_dim_2), ctypes.c_void_p(a_list.ctypes.data), ctypes.byref(theta_list_dim_1), ctypes.byref(theta_list_dim_2), ctypes.c_void_p(theta_list.ctypes.data), ctypes.byref(velocity_total_dim_1), ctypes.byref(velocity_total_dim_2), ctypes.byref(velocity_total))

    # Post-processing "velocity_total"
    velocity_total_size = (velocity_total_dim_1.value) * (velocity_total_dim_2.value)
    if (velocity_total_size > 0):
        velocity_total = numpy.array(ctypes.cast(velocity_total, ctypes.POINTER(ctypes.c_float*velocity_total_size)).contents, copy=False)
        velocity_total = velocity_total.reshape(velocity_total_dim_2.value,velocity_total_dim_1.value).T
    elif (velocity_total_size == 0):
        velocity_total = numpy.zeros(shape=(velocity_total_dim_2.value,velocity_total_dim_1.value), dtype=ctypes.c_float, order='F')
    else:
        velocity_total = None
    
    # Return final results, 'INTENT(OUT)' arguments only.
    return a_list, theta_list, velocity_total


# ----------------------------------------------
# Wrapper for the Fortran subroutine SENSOR_LINE_GENERATOR

def sensor_line_generator(x_boundary, nxf, pos_vector=None):
    ''''''
    
    # Setting up "x_boundary"
    if (type(x_boundary) is not ctypes.c_int): x_boundary = ctypes.c_int(x_boundary)
    
    # Setting up "nxf"
    if (type(nxf) is not ctypes.c_int): nxf = ctypes.c_int(nxf)
    
    # Setting up "pos_vector"
    if (pos_vector is None):
        pos_vector = numpy.zeros(shape=(3, nxf), dtype=ctypes.c_float, order='F')
    elif ((not issubclass(type(pos_vector), numpy.ndarray)) or
          (not numpy.asarray(pos_vector).flags.f_contiguous) or
          (not (pos_vector.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'pos_vector' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        pos_vector = numpy.asarray(pos_vector, dtype=ctypes.c_float, order='F')
    pos_vector_dim_1 = ctypes.c_long(pos_vector.shape[0])
    pos_vector_dim_2 = ctypes.c_long(pos_vector.shape[1])

    # Call C-accessible Fortran wrapper.
    clib.c_sensor_line_generator(ctypes.byref(x_boundary), ctypes.byref(nxf), ctypes.byref(pos_vector_dim_1), ctypes.byref(pos_vector_dim_2), ctypes.c_void_p(pos_vector.ctypes.data))

    # Return final results, 'INTENT(OUT)' arguments only.
    return pos_vector

