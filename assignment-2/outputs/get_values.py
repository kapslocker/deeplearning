import os

filename = 'vgg16_output.txt'
with open(filename, 'r') as myfile:
    i = 1
    for line in myfile:
        if i % 5 == 0:
            # print line
            print line.strip().split(':')[-1]
        i = i + 1
