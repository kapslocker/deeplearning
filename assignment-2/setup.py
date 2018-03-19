import os
''' Fetch videos from youtube using youtube-dl '''
comm = "youtube-dl -a videos_list.txt -o \'videos/%(title)s\'"
os.system(comm)
