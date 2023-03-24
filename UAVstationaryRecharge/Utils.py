import math


HEIGHT = 600



def getStartGoalCoords(maskaCoord):
    goal = -1
    for j in range(maskaCoord.shape[1]):
        if maskaCoord[0][j] == 0:
            start = j
            break
    for i in range(maskaCoord.shape[0]):
        for j in range(maskaCoord.shape[1]):
            if maskaCoord[i][j] == 0 and (j != start or i != 0):
                goal = j
                break
        if goal != -1:
            break
    start = (0, start)
    goal = (i, j)
    return start, goal

#
# def getStartGoalCoords(maskaCoord):
#     for j in range(maskaCoord.shape[1]):
#         if maskaCoord[0][j] == 0:
#             start = j
#             break
#     for j in range(maskaCoord.shape[1]):
#         if maskaCoord[maskaCoord.shape[0] - 1][j] == 0:
#             goal = j
#             break
#     return start, goal

def findHighestVertexCoord(coords):
    highest = 9000
    for i in range(len(coords)):
        highestLast = highest
        highest = min(highest, coords[i][1])
        if highestLast != highest:
            indexHigh = i

    return coords[indexHigh], indexHigh

def getAngleBy3Points(a, b, c):
    # последовательность точек строго по построенной линии (углу)
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def getAngle2By3Points (vertexes):
    ang = math.degrees(math.atan2(vertexes[2][1] - vertexes[1][1],
                                  vertexes[2][0] - vertexes[1][0]) - math.atan2(vertexes[0][1] - vertexes[1][1],
                                                                                vertexes[0][0] - vertexes[1][0]))
    return ang + 360 if ang < 0 else ang

def getNextLeftIndex (coords, indexCurrent):
    if indexCurrent == len(coords) - 1:
        index_next = 0
    else:
        index_next = indexCurrent + 1
    return index_next

def getNextRightIndex (coords, indexCurrent):
    if indexCurrent == 0:
        index_next = len(coords) - 1
    else:
        index_next = indexCurrent - 1
    return index_next

def find_k (p1, p2):
    k = (p2[1] - p1[1])/(p2[0] - p1[0])
    return k



def find_angle_OX(k):
    angle = math.degrees(math.atan(k))
    angle = - angle
    return angle


def find_angle_OY(k):
    angle = math.degrees(math.atan(k))
    angle = 90 + angle
    if angle > 90:
        angle = - (180 - angle)
    return angle

def coord_transform_to_NewCSx(angle_OY, x, y):
    x1 = y * math.sin(math.radians(angle_OY)) + x * math.cos(math.radians(angle_OY))
    y1 = y * math.cos(math.radians(angle_OY)) - x * math.sin(math.radians(angle_OY))
    return x1, y1

def coord_transform_to_NewCSx(angle_OY, tup):
    x1 = tup[1] * math.sin(math.radians(angle_OY)) + tup[0] * math.cos(math.radians(angle_OY))
    y1 = tup[1] * math.cos(math.radians(angle_OY)) - tup[0] * math.sin(math.radians(angle_OY))
    return (x1, y1)

# def coord_transform_to_OldCSx(angle_OY, x1, y1):
#     x = x1 * math.cos(math.radians(angle_OY)) - y1 * math.sin(math.radians(angle_OY))
#     y = y1 * math.cos(math.radians(angle_OY)) + x1 * math.sin(math.radians(angle_OY))
#     return x, y

def coord_transform_to_OldCSx(angle_OY, tup):
    x = tup[0] * math.cos(math.radians(angle_OY)) - tup[1] * math.sin(math.radians(angle_OY))
    y = tup[1] * math.cos(math.radians(angle_OY)) + tup[0] * math.sin(math.radians(angle_OY))
    return [x, y]

def coord_transform_to_OldCS_x(angle_OY, tup):
    x = tup[0] * math.cos(math.radians(angle_OY)) - tup[1] * math.sin(math.radians(angle_OY))
    y = tup[1] * math.cos(math.radians(angle_OY)) + tup[0] * math.sin(math.radians(angle_OY))
    return x

def coord_transform_to_OldCS_y(angle_OY, tup):
    x = tup[0] * math.cos(math.radians(angle_OY)) - tup[1] * math.sin(math.radians(angle_OY))
    y = tup[1] * math.cos(math.radians(angle_OY)) + tup[0] * math.sin(math.radians(angle_OY))
    return y












