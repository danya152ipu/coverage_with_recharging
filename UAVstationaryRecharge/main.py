import math
from scipy import ndimage
from Polygon import *
import pygame
import time
import os
import sys
import generator as gn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import Utils as ut
from coverageFromGit.coverage_test import *
import scipy.ndimage

pygame.init()
WIDTH = 800
HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 215, 0)
BLUE = (0,0,255)
wx = 20
wy = 20
BATTERY_PATH = 36000
kvadrSpeed = 1
clock = pygame.time.Clock()
dis = pygame.display.set_mode((WIDTH, HEIGHT))
game_over = False
pygame.font.init()
my_font = pygame.font.SysFont('Comic Sans MS', 30)
since = time.perf_counter()

coords = gn.draw_polygon(4)
highestVertexCoord = ut.findHighestVertexCoord(coords)

def transform(
        grid_map, src, distance_type='chessboard',
        transform_type='path', alpha=0.01
):
    n_rows, n_cols = grid_map.shape
    if n_rows == 0 or n_cols == 0:
        sys.exit('Empty grid_map.')
    inc_order = [[0, 1], [1, 1], [1, 0], [1, -1],
                 [0, -1], [-1, -1], [-1, 0], [-1, 1]]
    if distance_type == 'chessboard':
        cost = [1, 1, 1, 1, 1, 1, 1, 1]
    elif distance_type == 'eculidean':
        cost = [1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2)]
    else:
        sys.exit('Unsupported distance type.')
    transform_matrix = float('inf') * np.ones_like(grid_map, dtype=float)
    transform_matrix[src[0], src[1]] = 0
    if transform_type == 'distance':
        eT = np.zeros_like(grid_map)
    elif transform_type == 'path':
        eT = ndimage.distance_transform_cdt(1 - grid_map, distance_type)
    else:
        sys.exit('Unsupported transform type.')

    # set obstacle transform_matrix value to infinity
    for i in range(n_rows):
        for j in range(n_cols):
            if grid_map[i][j] == 1.0:
                transform_matrix[i][j] = float('inf')
    is_visited = np.zeros_like(transform_matrix, dtype=bool)
    is_visited[src[0], src[1]] = True
    traversal_queue = [src]
    calculated = [(src[0] - 1) * n_cols + src[1]]

    def is_valid_neighbor(g_i, g_j):
        return 0 <= g_i < n_rows and 0 <= g_j < n_cols \
            and not grid_map[g_i][g_j]

    while traversal_queue:
        i, j = traversal_queue.pop(0)
        for k, inc in enumerate(inc_order):
            ni = i + inc[0]
            nj = j + inc[1]
            if is_valid_neighbor(ni, nj):
                is_visited[i][j] = True

                # update transform_matrix
                transform_matrix[i][j] = min(
                    transform_matrix[i][j],
                    transform_matrix[ni][nj] + cost[k] + alpha * eT[ni][nj])

                if not is_visited[ni][nj] \
                        and ((ni - 1) * n_cols + nj) not in calculated:
                    traversal_queue.append((ni, nj))
                    calculated.append((ni - 1) * n_cols + nj)
    return transform_matrix

def get_search_order_increment(start, goal):
    if start[0] >= goal[0] and start[1] >= goal[1]:
        order = [[1, 0], [0, 1], [-1, 0], [0, -1],
                 [1, 1], [1, -1], [-1, 1], [-1, -1]]
    elif start[0] <= goal[0] and start[1] >= goal[1]:
        order = [[-1, 0], [0, 1], [1, 0], [0, -1],
                 [-1, 1], [-1, -1], [1, 1], [1, -1]]
    elif start[0] >= goal[0] and start[1] <= goal[1]:
        order = [[1, 0], [0, -1], [-1, 0], [0, 1],
                 [1, -1], [-1, -1], [1, 1], [-1, 1]]
    elif start[0] <= goal[0] and start[1] <= goal[1]:
        order = [[-1, 0], [0, -1], [0, 1], [1, 0],
                 [-1, -1], [-1, 1], [1, -1], [1, 1]]
    else:
        sys.exit('get_search_order_increment: cannot determine \
              start=>goal increment order')
    return order

def wavefront(transform_matrix, start, goal):
    path = []
    n_rows, n_cols = transform_matrix.shape

    def is_valid_neighbor(g_i, g_j):
        is_i_valid_bounded = 0 <= g_i < n_rows
        is_j_valid_bounded = 0 <= g_j < n_cols
        if is_i_valid_bounded and is_j_valid_bounded:
            return not is_visited[g_i][g_j] and \
                transform_matrix[g_i][g_j] != float('inf')
        return False

    inc_order = get_search_order_increment(start, goal)
    current_node = start
    is_visited = np.zeros_like(transform_matrix, dtype=bool)
    while current_node != goal:
        i, j = current_node
        path.append((i, j))
        is_visited[i][j] = True
        max_T = float('-inf')
        i_max = (-1, -1)
        i_last = 0
        for i_last in range(len(path)):
            current_node = path[-1 - i_last]  # get latest node in path
            for ci, cj in inc_order:
                ni, nj = current_node[0] + ci, current_node[1] + cj
                if is_valid_neighbor(ni, nj) and \
                        transform_matrix[ni][nj] > max_T:
                    i_max = (ni, nj)
                    max_T = transform_matrix[ni][nj]
            if i_max != (-1, -1):
                break
        if i_max == (-1, -1):
            break
        else:
            current_node = i_max
    path.append(goal)
    return path

def get_path(points, line, yForStr, maskaCoord):
    maskaCoord = np.array(maskaCoord)
    img = maskaCoord.copy()
    start, goal = ut.getStartGoalCoords(maskaCoord)
    new_row_to_add = []
    for i in range(maskaCoord.shape[1]):
        if i == start[1]:
            new_row_to_add.append(0)
        else:
            new_row_to_add.append(1)
    new_row_to_add = np.array(new_row_to_add)
    maskaCoord = np.vstack((new_row_to_add, maskaCoord))
    new_row_to_points = points[0]
    points = np.vstack((new_row_to_points, points))
    new_row_to_yForStr = yForStr[0]
    points = np.vstack((new_row_to_yForStr, yForStr))
    start = (start[0] + 1, start[1])
    goal = (0, start[1])
    DT = transform(img, goal)
    DT_path = wavefront(DT, start, goal)
    return DT_path

def draw_path(tr_coords1, tr_coords2, highest_vertex, wx, wy, dis, color):

    points, line, yForStr, maskaCoord = Polygon2.create_grid_for_triangle2(tr_coords1, tr_coords2,
                                                                           highest_vertex, wx, wy, dis, color)
    # вычисление траектории покрытия и ее отрисовка
    sm_x = 0
    sm_y = 0
    k = 0
    bol = False
    DT_path = get_path(points, line, yForStr, maskaCoord)
    path_cost = 0

    while k < len(DT_path) - 1:
        clock.tick(70)
        x1 = points[DT_path[k][0], DT_path[k][1]]
        y1 = yForStr[DT_path[k][0], DT_path[k][1]]
        x1_next = points[DT_path[k + 1][0], DT_path[k + 1][1]]
        y1_next = yForStr[DT_path[k + 1][0], DT_path[k + 1][1]]
        path_cost += math.hypot(x1_next - x1, y1_next - y1)
        if not bol:
            bol = True
            beg = [x1, y1]
            dx = x1_next - beg[0]
            dy = y1_next - beg[1]
            rads = math.atan2(dy, dx)
            degs = math.degrees(rads)
        if math.hypot(beg[0] - x1_next, beg[1] - y1_next) >= kvadrSpeed:
            beg[0] = beg[0] + sm_x
            beg[1] = beg[1] + sm_y
            pygame.draw.circle(dis, BLUE, beg, 2)
            pygame.display.update()
            sm_x = kvadrSpeed * math.cos(rads)
            sm_y = kvadrSpeed * math.sin(rads)
            pygame.display.update()
        else:
            bol = False

        if not bol:
            if k <= len(DT_path) - 1:
                k += 1
            sm_x = 0
            sm_y = 0
    return path_cost

area = Polygon2(coords, dis)
opt_triangle_coords = area.findOptimalTriangle(6000, dis)
highestVertexCoord = highestVertexCoord[0]
m = 0


while not game_over:
    clock.tick(1)
    pygame.display.set_caption('Планирование траектории')
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
    if m == 0:
        dis.fill(WHITE)
        area.drawPolygon(BLACK)
        # area.drawRectangularGrid(wx, wy)
        color_1param = 0
        color_3param = 0
        color_2param = 255
        for i in range(len(opt_triangle_coords) - 1):
            if i % 2 == 0:
                color_1param = 150
            else:
                color_1param = 0
            color = [color_1param, color_2param, color_3param]
            pygame.display.update()
            color_3param += 22
            color_2param -= 23
        for i in range(len(opt_triangle_coords)):
            pygame.draw.line(dis, BLACK, highestVertexCoord, opt_triangle_coords[i], 5)
        for i in range(len(opt_triangle_coords) - 1):
            path_cost = draw_path(opt_triangle_coords[i], opt_triangle_coords[i + 1],
                      highestVertexCoord, wx, wy, dis, tuple(color))
            print(f'{i + 1} path = {path_cost}')
        m = 1
    # for i in range(len(opt_triangle_coords)):
    #     pygame.draw.line(dis, (0, 0, 0), highestVertexCoord, opt_triangle_coords[i], 5)
    pygame.display.update()





