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

def run():

    # example_first()

    # array_creation()

    # basic_operations()

    # indexing_slicing_iterating()

    shape_manipulation()

    return 0

def main():

    rc = run()
    

    return 0

main()
