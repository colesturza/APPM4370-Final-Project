import c3d #mocap file reading library, could be better ones
import numpy as np

def load_c3d(file):
    with open(file, 'r') as data:
        reader = c3d.Reader(data)

        for i, (points, analog) in enumerate(reader.read_frames()):
            #i is the fram index
            #points is np array of point data
            #analog is np array of analog data
            pass
