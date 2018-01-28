import os, struct
import numpy as np


def read_data(choice = 0):
    ''' return label, 2D image array
        choice: 0 -> train
                1 -> test
    '''
    if choice==0:
        img_loc = "byclass/train_images"
        label_loc = "byclass/train_labels"
    else:
        img_loc = "byclass/test_images"
        label_loc = "byclass/test_labels"
    # unpack ubyte fileloc given as C struct
    with open(label_loc, 'rb') as labels:
        fmt, num = struct.unpack(">II", labels.read(8))
        lbl = np.fromfile(labels, dtype=np.int8)

    count = len(lbl)
    with open(img_loc, 'rb') as images:
        fmt, num, rows, cols = struct.unpack(">IIII", images.read(16))
        img = np.fromfile(images, dtype = np.uint8).reshape(count, rows, cols)
    for i in xrange(count):
        yield (lbl[i], img[i])

def display(npimage):
    ''' Display a numpy 2D array'''
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(npimage, cmap = mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


training_data = list(read_data(0))

label, pixels = training_data[3]
