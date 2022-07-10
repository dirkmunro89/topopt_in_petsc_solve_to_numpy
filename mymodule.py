import numpy

def array_tutorial(a):
    print("array_tutorial - python")
    print(a)
    print(numpy.dtype(a[0,0]))
    print("")
    firstRow = a[0,:]
    return firstRow

def myfunction():
    beta = numpy.array([[1,2,3],[1,2,3],[1,2,3]],dtype=numpy.float128)
    print("myfunction - python")
    print(beta)
    print("")
    firstRow = beta[0,:]
    return firstRow
