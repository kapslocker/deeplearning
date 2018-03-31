import os

dir = "videos/"

for category in os.listdir(dir):
    for item in os.listdir(dir + category):
        os.system("python label_data.py \"{0}\" {1}".format(dir + category + "/" + item, category))
