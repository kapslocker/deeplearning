import os, struct
import numpy as np
from itertools import imap

def get_mapping():
    ''' (ASCII, label) '''
    with open("byclass/emnist-byclass-mapping.txt") as mapping:
        rows = imap(str.split, mapping)
        data = [(int(row[1]), int(row[0])) for row in rows]
    return dict(data)

def read_data(choice, labelstring):
    ''' return label, 2D image array
        choice: 0 -> train
                1 -> test
        labelstring: letters to take. e.g. "abcdefghi" (NOTE: Do not duplicate letters)
    '''
    mp = get_mapping()
    labels = [mp[ord(ch)] for ch in labelstring]
    if choice==0:
        img_loc = "byclass/train_images"
        label_loc = "byclass/train_labels"
    else:
        img_loc = "byclass/test_images"
        label_loc = "byclass/test_labels"
    # unpack ubyte fileloc given as C struct
    with open(label_loc, 'rb') as labelfile:
        fmt, num = struct.unpack(">II", labelfile.read(8))
        lbl = np.fromfile(labelfile, dtype=np.int8)
    count = len(lbl)
    with open(img_loc, 'rb') as images:
        fmt, num, rows, cols = struct.unpack(">IIII", images.read(16))
        img = np.fromfile(images, dtype = np.uint8).reshape(count, rows * cols, 1)
    for i in xrange(count):
        if lbl[i] in labels:
            yield (img[i], vector(labels.index(lbl[i]), len(labels)))

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

def vector(index, num_classes):
    a = np.zeros((num_classes, 1))
    a[index] = 1.0
    return a

def train_data(labelstring):
     return list(read_data(0, labelstring))

def test_data(labelstring):
    return list(read_data(1, labelstring))

# labelstring = "abcdefghi"
# m = test_data(labelstring)
# label, img = m[0]
# print label
# display(img)
