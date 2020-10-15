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


def run():

    example_first()

    array_creation()


    return 0

def main():

    rc = run()
    

    return 0

main()
