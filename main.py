import numpy as np

def example_first():
    a = np.arange(15).reshape(3, 5)

    print('a is:\n', a)

def array_creation():

    python_array = [x ** 2 for x in range(100)]
    np_array = np.array(python_array)

    print('python array: ', python_array)
    print('numpy array: ', np_array)

    # common mistake is forgeting that numpy np.array converts sequences of sequences into 2D arrays
    # and sequences of sequences of sequences into 3D arrays

    np_array_2d_v1 = np.array([[3, 4, 5], [5, 5, 5]])
    np_array_2d_v2 = np.array([(3, 4, 5), (5, 5, 5)])

    print('np2dv1:\n', np_array_2d_v1)
    print('np2dv2:\n', np_array_2d_v2)

    print('equal v1 and v2: ', np_array_2d_v1 == np_array_2d_v2)

    np_array_3d = np.array([[[2, 2], [1, 2], [2, 2]], [[1, 2], [2, 2]]], dtype=object)
    print('np3d:\n', np_array_3d)

    np_array_complex = np.array([ [1, 2], [3, 4] ], dtype=complex)

    print('np array complex:\n', np_array_complex)

    # initialize arrays

    np_zeros = np.zeros((5, 15))
    print('np_zeros:\n', np_zeros)

    np_ones = np.ones((3, 4))
    print('np_ones:\n', np_ones)

    np_empty = np.empty( (3, 2) )
    print('np_empty:\n', np_empty)

    # create sequences of numbers (similar like pythons range())

    np_sequence = np.arange(10, 45, 3)
    print('np_sequence:\n', np_sequence)

    np_sequence_float = np.arange(0.1, 3.8, 0.05)
    print('np_sequence_float:\n', np_sequence_float)

    # np.arange is best avoided with float numbers because finite precision of float numbers
    # in np it is considered best practice to use np.linspace for float numbers
    # it is not provided step, rather, it is provided start, end and number of elements to fit into it

    np_linspace = np.linspace(0, 2, 27)
    print('np_linspace:\n', np_linspace)

    np_linspace_pi = np.linspace(0, 2 * np.pi, 100)
    np_sin = np.sin(np_linspace_pi)

    print('np_sin:\n', np_sin) # np.sin(x) x takes array-like for input and calculates sin on each element

    # see also
    # array, zeros, zeros_like, ones, ones_like, empty, empty_like, arange, linspace, numpy.random.Generator.rand, numpy.random.Generator.randn, fromfunction, fromfile

    return 0

def basic_operations():

    a = np.array( [20, 30, 40, 50])
    b = np.arange(4)

    c = a - b

    print('a - b = ', c)

    b_sqr = b ** 2

    print('b ** 2 = ', b_sqr)

    np_const_times_array = 10 * np.tan(a)

    print('c * array = ', np_const_times_array)

    np_bool_on_array = np_const_times_array > -10.0

    print('bool condition array = ', np_bool_on_array)

    # unlike in many other matrix languages product operator * operates elementwise in np arrays
    # matrix product in mathematical context is achieved with @ (python >= 3.5) or the dot function

    A = np.array( [ [1, 1], [0, 1] ] )
    B = np.array( [ [2, 3], [1, 2] ] )

    C = A * B

    print('elementwise product is =\n', C)

    D = A @ B

    print('matrix product is =\n',  D)

    F = A.dot(B)

    print('also matrix product =\n ', F)

    rg = np.random.default_rng(1)
    a = np.ones((2, 3), dtype=int)
    b = rg.random( (2, 3) )

    # a *= b
    # print('a *= b = ', a)

    b += a

    print('b += a = ', b)

    # a += b

    # print('a += b = ', a) # numpy.core._exceptions.UFuncTypeError: Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'

    # upcasting => when operating on arrays with different types corresponding array is one that is superset to other

    a = np.ones(3, dtype=np.int32)
    b = np.linspace(0, np.pi, 3)

    print(' a.size == b.size = ', a.size == b.size)

    c = a + b

    print(' a + b = ', c)

    print(' c type: ', c.dtype)

    d = np.exp(c * 1j)

    print(' d = exp(complex times c) = ', d)

    # unary operations on arrays

    a = rg.random( (3, 5) )
    a_sum = a.sum()

    print(' a.sum = ', a_sum)

    a_min = a.min()

    print('min element in a: ', a_min)

    a_max = a.max()

    print('max element in a: ', a_max)

    # unary operations can be called upon one axis

    # b = np.arange(15).shape(5, 3)
    b = np.arange(15).reshape(5, 3)

    b_sum = b.sum(axis = 0)

    b_min = b.min(axis = 1)

    b_max = b.max(axis = 0)

    print('b matrix:\n', b)
    print('sum along 0: ', b_sum, ' min along 1: ', b_min, ' max along 1: ', b_max)

    b_cumsum = b.cumsum(axis=1)

    print('b_cumsum on axis = 1:\n', b_cumsum)

    # universal functions 
    # numpy provides familiar mathematical functions called ufunc
    # ufuncs operate elementwise on an array thus producing array as output

    B = np.arange(3)
    print('B = ', B)
    print('exp = ', np.exp(B))
    print('sqrp = ', np.sqrt(B))
    print('np.add = ', np.add(B, np.exp(B)))

    # see also
    # all, any, apply_along_axis, argmax, argmin, argsort, average, bincount, ceil, clip, conj, corrcoef, cov, cross, cumprod, cumsum, diff, dot, floor, inner, invert, lexsort, max, maximum, mean, median, min, minimum, nonzero, outer, prod, re, round, sort, std, sum, trace, transpose, var, vdot, vectorize, where
    

def run():

    # example_first()

    # array_creation()

    basic_operations()


    return 0

def main():

    rc = run()
    

    return 0

main()
