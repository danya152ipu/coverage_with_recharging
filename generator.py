import numpy as np
from matplotlib import pyplot as plt

def draw_polygon(n):

    x = np.random.randint(0,500,n)
    y = np.random.randint(0,500,n)
    #computing center point of the polygon
    center_point = [np.sum(x)/n, np.sum(y)/n]
    angles = np.arctan2(x-center_point[0],y-center_point[1])

    #sorting the points:
    sort_tups = sorted([(i,j,k) for i,j,k in zip(x,y,angles)], key = lambda t: t[2])
    print(sort_tups)

    #check that no duplicates:
    if len(sort_tups) != len(set(sort_tups)):
        raise Exception('two equal coordinates -- exiting')

    x,y,angles = zip(*sort_tups)
    x = list(x)
    y = list(y)
    coords = list(zip(x,y))
    return coords

def generatePolygons(vertex, number):
    draw_polygon(vertex)