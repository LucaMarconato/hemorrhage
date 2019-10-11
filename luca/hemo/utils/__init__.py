import numpy

def clip_normalize(a, mi, ma):
    retval = numpy.clip(a, mi, ma)
    retval -= mi 
    retval /= (ma - mi)
    return retval


def unconcatenate(x, sizes):
    last = 0
    splitted = []
    for size in sizes:
        a = x[last:last+size, :]
        splitted.append(a)
        last += size

    return splitted
