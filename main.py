import numpy as np
import time

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

def indexing_slicing_iterating():
    a = np.arange(10) ** 3

    print('a arange ** 3: ', a)

    print(' a[2] = ', a[2])

    print(' a[2:5] = ', a[2:5])

    # equivalent to a[0:6:2] = 10000
    # from start to position 6, set every 2nd element to 10000

    a[:6:2] = 10000

    print(' a[:6:2] = 10000 = ', a) 

    a_reversed = a[ : : -1]

    print('a_reversed = ', a_reversed)

    for i in a_reversed:
        print('i ** (1/3.) = ', i ** (1/3.))

    # multidimensional arrays can have one index per axis

    def f(x, y):
        return 10 * x + y

    b = np.fromfunction(f, (5, 5), dtype=np.int64)

    print('b fromfunction =\n', b)

    print('direct indexing: ', b[2, 3])

    # start : end : step
    
    print('each row:\n', b[:5, :]) # of course it is same as matrix b
    print('first 4 rows:\n', b[:4, :])
    print('every second column:\n', b[:, ::2])
    print('each row in second column: ', b[:5, 1])

    # iteration is done with respect to the first axis
    for row in b:
        print('row => ', row)
    
    # if one wants to iterate through every element of multidimensional array => array.flat
    for element in b.flat:
        print('element => ', element)

    # see also
    # Indexing, Indexing (reference), newaxis, ndenumerate, indices

def shape_manipulation():

    rg = np.random.default_rng(1)
    a = np.floor( 10 * rg.random( (3, 4) ))

    print('shape of a: ', a.shape)

    # ndarray.ravel(), ndarray.reshape(), ndarray.T return modified array but don't change the original

    print('a: \n', a)
    
    a_flattened = a.ravel()

    print('a_flattened: \n', a_flattened)

    a_reshaped = a.reshape(2, 6)

    print('a_reshaped: \n', a_reshaped)

    a_transposed = a.T
    
    print('a_transposed: \n', a_transposed)

    # ndarray.resize() modifies the array itself

    print('a: \n', a)
    a.resize( (2, 6) )
    print('a_resized:\n', a)

    # stacking together different arrays

    a = np.floor(10 * rg.random( (2, 2) ))

    print('a: \n', a)

    b = np.floor(10 * rg.random( (2, 2) ))

    print('b: \n', b)

    # vertical stack

    v_stack = np.vstack( (a, b) )
    h_stack = np.hstack( (a, b) )

    print('vertical stack: \n', v_stack, '\nhorizontal stack: \n', h_stack)
    
    v_stack_multiple = np.vstack( (a, b, a, b) )
    
    print('vertical stack multiple: \n', v_stack_multiple)

    # column_stack stacks 1d arrays as columns into a 2d array

    cs_2d = np.column_stack( (a, b) )

    print('column stack with 2d arrays: \n', cs_2d)

    a = np.array( [1, 2, 3, 4, 5] )
    b = np.array( [2, 3, 4, 1, 6] )

    c = np.column_stack( (a, b) )

    print('column stack from 2 1d arrays: \n', c)

    c_hstack = np.hstack( (a, b) )

    print('c_hstack: ', c_hstack)

    a_column_vector = a[:, np.newaxis]
    b_column_vector = b[:, np.newaxis]

    ab_column_stack = np.column_stack( (a_column_vector, b_column_vector) )

    ab_hstack = np.hstack( (a_column_vector, b_column_vector) )

    print('ab_column_stack: \n', ab_column_stack, '\nab_hstack: \n', ab_hstack)    

    # hstack and column_stack are different
    # vstack and row_stack are equivalent

    print('np.column_stack is np.hstack: ', np.column_stack is np.hstack)
    print('np.row_stack is np.vstack', np.row_stack is np.vstack)

    # concatenate allows for an optional argument giving the number of the axis along which the concatenation should happen

    # stack numbers along one axis

    rr = np.r_[1:16, 2, 3, 5]

    print('stacked numbers on one axis: ', rr)

    # see also
    # hstack, vstack, column_stack, concatenate, c_, r_

    # splitting arrays

    a = np.floor(10 * rg.random( (2, 12) ))

    print('a: \n', a)

    # split a in 3
    print('a splitted:\n', np.hsplit(a, 3))

    print('a split after the third and the fourth column: \n', np.hsplit(a, (3, 4)))

def copies_and_views():

    a = np.array( [[0, 1, 2, 3],
                   [4, 5, 6, 7],
                   [8, 9, 10, 11]])
    print('a: \n', a)

    b = a

    print('b is a: ', b is a)

    # view or shallow copy
    # the view method creates a new array object that looks at the same data

    c = a.view()

    print('c is a: ', c is a)
    print('c.base is a: ', c.base is a)
    print('c.flags.owndata: ', c.flags.owndata)

    c = c.reshape( (2, 6) )

    print('a.shape: ', a.shape)

    c[0, 4] = 4096

    print('a: \n', a)
    print('c: \n', c)

    # slicing an array returns a view of it
    s = a[ :, 1:3]

    print('s: \n', s)

    s[:] = 123

    print('a: \n', a)

    # deep copy
    # copy method makes a complete copy of the array and its data

    d = a.copy()

    print('d is a: ', d is a)
    print('d.base is a: ', d.base is a)

    d[0, 0] = 12345

    print('a: \n', a, '\nd: \n', d)

    # sometimes copy should be used when slicing, example: large intermidiate result sliced

    a = np.arange(int(1e8))

    b = a[::55].copy()

    del a

    print('b: \n', b) 

def functions_and_methods_array_creation():
    
    # https://numpy.org/devdocs/reference/generated/numpy.arange.html#numpy.arange
    x = np.arange(1, 1234, 21, dtype=np.int64)

    print('np.arange = ', x) # np.arange works like python range

    # https://numpy.org/devdocs/reference/generated/numpy.array.html#numpy.array
    x = np.array( [1, 2, 3, 4, 5, 6] )

    print('np.array = ', x)


    # https://numpy.org/devdocs/reference/generated/numpy.copy.html#numpy.copy
    x = np.array([1, 2, 3])

    y = x

    z = np.copy(x)

    x[0] = 17
    print('x: ', x, ' y: ', y, ' z: ', z)

    # https://numpy.org/devdocs/reference/generated/numpy.empty.html#numpy.empty
    x = np.empty( (3, 4) )

    print('x: \n', x)

    x = np.empty( (2, 2), dtype=np.int)

    print('x: \n', x)

    # https://numpy.org/devdocs/reference/generated/numpy.empty_like.html#numpy.empty_like
    x = np.array( [[1, 2, 3], [2, 3, 4]] )
    y = np.empty_like(x)

    print('x: \n', x, '\ny: \n', y)


    x = np.array( [[1., 2., 3.], [2., 3., 4.]] )
    y = np.empty_like(x)

    print('x: \n', x, '\ny: \n', y)

    # https://numpy.org/devdocs/reference/generated/numpy.eye.html#numpy.eye
    x = np.eye(7, dtype=np.int64)

    print('x: \n', x)


    x = np.eye(7, 4, dtype=np.float64)

    print('x: \n', x)

    # https://numpy.org/devdocs/reference/generated/numpy.fromfile.html#numpy.fromfile
    # later

    # https://numpy.org/devdocs/reference/generated/numpy.fromfunction.html#numpy.fromfunction
    def f(x, y):
        return (x**2 - y)*0.2
    x = np.fromfunction(f, (4, 7), dtype=np.float64)

    print('x: \n', x)

    x = np.fromfunction(lambda i, j: i + 2 * j, (5, 7), dtype=np.int64)

    print('x: \n', x)

    # https://numpy.org/devdocs/reference/generated/numpy.identity.html#numpy.identity
    x = np.identity(6)

    print('x: \n', x)

    x = np.identity(4, dtype=np.float32)

    print('x: \n', x)

    # https://numpy.org/devdocs/reference/generated/numpy.linspace.html#numpy.linspace
    x = np.linspace(2.0, np.pi * 6, num=23)

    print('x: \n', x)
    
    x = np.linspace(2.0, np.pi * 6, num=23, endpoint=False)

    print('x: \n', x)

    x = np.linspace(2.0, np.pi * 6, num=23, retstep=True)

    print('x: \n', x[0], '\nwith step:\n', x[1])

    # https://numpy.org/devdocs/reference/generated/numpy.logspace.html#numpy.logspace
    x = np.logspace(2.0, 4.0, num=6)

    print('x: \n', x)

    x = np.logspace(2.0, 4.0, num=6, endpoint=False)

    print('x: \n', x)

    x = np.logspace(2.0, 4.0, num=6, base=2.0)

    print('x: \n', x)
    # note for logspace
    # In linear space, the sequence starts at base ** start (base to the power of start) and ends with base ** stop (see endpoint below).

    # https://numpy.org/devdocs/reference/generated/numpy.mgrid.html#numpy.mgrid
    # https://numpy.org/devdocs/reference/generated/numpy.ogrid.html#numpy.ogrid
    # later

    # https://numpy.org/devdocs/reference/generated/numpy.ones.html#numpy.ones
    x = np.ones( (2, 7), dtype=np.int32)

    print('x: \n', x)

    s = (4, 4)
    x = np.ones( s, dtype=np.int32)

    print('x: \n', x)

    # https://numpy.org/devdocs/reference/generated/numpy.ones_like.html#numpy.ones_like
    x = np.arange(18)
    x.resize( (3, 6) )

    print('x: \n', x)

    x = np.ones_like(x) 

    print('x: \n', x)

    # https://numpy.org/devdocs/reference/generated/numpy.r_.html#numpy.r_
    x = np.r_[np.array([1, 2, 3]), [1]*4, np.array([1, 2])]

    print('x: \n', x)

    y = np.array( [[1, 2, 3], [4, 5, 6]] )
    x = np.r_['-1', y, y]

    print('x: \n', x)

    x = np.r_['0', y, y]

    print('x: \n', x)

    x = np.r_['r', [1, 2, 3], [1, 2, 3, 4]]
    
    print('x: \n', x)

    # https://numpy.org/devdocs/reference/generated/numpy.zeros.html#numpy.zeros
    x = np.zeros( (3, 6), dtype=np.float64)

    print('x: \n', x)

    # https://numpy.org/devdocs/reference/generated/numpy.zeros_like.html#numpy.zeros_like
    y = np.arange(25).reshape( (5, 5) )
    x = np.zeros_like(y)

    print('y: \n', y)
    print('x: \n', x)

def functions_and_methods_array_conversions():

    # https://numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html#numpy.ndarray.astype
    x = np.arange(24, dtype=np.int64).reshape(4, 6)
    y = x.astype(dtype=np.float64)

    print('x: \n', x, '\ny: \n', y)

    # https://numpy.org/devdocs/reference/generated/numpy.atleast_1d.html#numpy.atleast_1d
    x = 5.9
    y = np.atleast_1d(x)

    print('y: \n', y)

    # note: higher dimensionalities are preserved
    x = np.arange(25).reshape( (5, 5) )
    y = np.atleast_1d(x)

    print('x: \n', x, '\ny: \n', y)

    # https://numpy.org/devdocs/reference/generated/numpy.atleast_2d.html#numpy.atleast_2d
    x = 5.9
    y = np.atleast_2d(x)

    print('y: \n', y)
    # also preserves higher dimensionalities

    # https://numpy.org/devdocs/reference/generated/numpy.atleast_3d.html#numpy.atleast_3d
    x = 5.9
    y = np.atleast_3d(x)

    print('y: \n', y)

    # https://numpy.org/devdocs/reference/generated/numpy.mat.html#numpy.mat
    x = np.array( [[1, 2], [3, 4]] )
    m = np.asmatrix(x)

    print('x: \n', x, '\nm: \n', m)

    x = np.array( [1, 2] )
    m = np.asmatrix(x)

    print('x: \n', x, '\nm: \n', m)


def run():

    example_first()

    array_creation()

    basic_operations()

    indexing_slicing_iterating()

    shape_manipulation()

    copies_and_views()

    functions_and_methods_array_creation()

    functions_and_methods_array_conversions()

    return 0

def main():

    now = time.time()

    rc = run()

    print('function returned: ', rc, ', and it took: ', time.time() - now, ' seconds to finish')    

    return 0

main()
