import math
from Polygon import *
import pygame
from coverageFromGit.coverage_test import *
import time
import os
import sys
import generator as gn
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

pygame.init()
WIDTH = 800
HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 215, 0)
BLUE = (0,0,255)
pygame.display.set_caption("flyby")
FPS = 15
radiusField = 15
BATTERY_PATH = 550
clock = pygame.time.Clock()
kvadrSpeed = 1
# speedUGV = 0.5
coords = []
wx = 20 # ширина углового поля
wy = 20  # высота углового поля

dis = pygame.display.set_mode((WIDTH, HEIGHT))
game_over = False
pygame.font.init()
my_font = pygame.font.SysFont('Comic Sans MS', 30)
since = time.perf_counter()
do_animation = True

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
    transform_matrix = float('inf') * np.ones_like(grid_map, dtype = float)
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

def visualize_path(grid_map, start, goal, path):
    oy, ox = start
    gy, gx = goal
    # px, py = np.transpose(np.flipud(np.fliplr(path)))
    px,py = np.transpose(np.fliplr(path))
    if not do_animation:
        plt.imshow(grid_map, cmap='Greys')
        plt.plot(ox, oy, "-xy")
        plt.plot(px, py, "-r")
        plt.plot(gx, gy, "-pg")
        plt.show()
    else:
        for ipx, ipy in zip(px, py):
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.imshow(grid_map, cmap='Greys')
            plt.plot(ox, oy, "-xb")
            plt.plot(px, py, "-r")
            plt.plot(gx, gy, "-pg")
            plt.plot(ipx, ipy, "or")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.1)

def getStartGoalCoords(maskaCoord):
    for j in range(maskaCoord.shape[1]):
        if maskaCoord[0][j] == 0:
            start = j
            break
    for j in range(maskaCoord.shape[1]):
        if maskaCoord[maskaCoord.shape[0] - 1][j] == 0:
            goal = j
            break
    return start, goal

path_cost_to_recharge = []
coords_recharge2 = []
def getRechargeCoords(DT_path):
    path_cost = 0
    PATH_COST = 0
    recharge = []
    add_list = []
    for k in range(len(DT_path) - 1):
        x1 = points[DT_path[k][0], DT_path[k][1]]
        y1 = yForStr[DT_path[k][0]]
        x1_next = points[DT_path[k + 1][0], DT_path[k + 1][1]]
        y1_next = yForStr[DT_path[k + 1][0]]
        PATH_COST += math.hypot(x1_next - x1, y1_next - y1)
        path_cost += math.hypot(x1_next - x1, y1_next - y1)
        if k == 0:
            coords_recharge2.append((x1, y1))
        if path_cost >= BATTERY_PATH:
            recharge.append(DT_path[k])
            path_cost_to_recharge.append(path_cost - math.hypot(x1_next - x1, y1_next - y1))
            path_cost = math.hypot(x1_next - x1, y1_next - y1)
            coords_recharge2.append((x1, y1))
    return recharge, PATH_COST

coords = gn.draw_polygon(5)
area = Polygon2(coords, dis)
points, line, yForStr, maskaCoord = area.drawRectangularGrid(wx, wy)
maskaCoord = np.array(maskaCoord)
maskaCoord = np.reshape(maskaCoord, (line, int(maskaCoord.shape[0] / line)))
img = maskaCoord.copy()
start, goal = getStartGoalCoords(maskaCoord)
points = np.array(points)
points = np.reshape(points, (line, int(points.shape[0] / line)))
yForStr = np.linspace(yForStr, yForStr + line * wy, line + 1)
goal = (maskaCoord.shape[0] - 1, goal)
start = (0, start)
DT = transform(img, goal)
DT_path = wavefront(DT, start, goal)
recharge = []
recharge, PATH_COST = getRechargeCoords(DT_path)
n = 0
m = 0
k = 0
k1 = 0
sm_x = 0
sm_y = 0
path_coords = []
coords_recharge= []
bol = False
bol2 = False
UGV_velocity = []
time = []
time.append(0)
time_counter = 0
recgarge_time = 30
while not game_over:
    clock.tick(100)
    pygame.display.set_caption('Планирование траектории')
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
    x1 = points[DT_path[k][0], DT_path[k][1]]
    y1 = yForStr[DT_path[k][0]]
    x1_next = points[DT_path[k+1][0], DT_path[k+1][1]]
    y1_next = yForStr[DT_path[k+1][0]]
    if m == 0:
        startForUGV = (x1,y1)
        dis.fill(WHITE)
        area.drawPolygon(BLACK)
        area.drawRectangularGrid(wx, wy)
        pygame.display.update()
        text = my_font.render(f' {PATH_COST} sec', False, (0, 0, 0))
        dis.blit(text, (500, 500))
        coords_recharge.append((x1,y1))
        m = 1
        for step in coords_recharge2:
            pygame.draw.rect(dis, YELLOW, (step[0] - wx / 2, step[1] - wy / 2, wx, wy), 6)
    text = my_font.render(f' {PATH_COST} sec', False, (0, 0, 0))
    dis.blit(text, (500, 500))

    if not bol:
        bol = True
        beg = [x1,y1]
        dx = x1_next - beg[0]
        dy = y1_next - beg[1]
        rads = math.atan2(dy, dx)
        degs = math.degrees(rads)
    if math.hypot(beg[0] - x1_next, beg[1] - y1_next) >= kvadrSpeed:
        beg[0] = beg[0] + sm_x
        beg[1] = beg[1] + sm_y
        pygame.draw.circle(dis,BLACK,beg,2)
        sm_x = kvadrSpeed * math.cos(rads)
        sm_y = kvadrSpeed * math.sin(rads)
        pygame.display.update()
    else:
        bol = False

    if not bol:
        if k < len(DT_path) - 2:
            k += 1
        sm_x = 0
        sm_y = 0

# creating recharge lines
    if k1 <= len(coords_recharge2)  - 2:
        if not bol2:
            sm_x1 = 0
            sm_y1 = 0
            bol2 = True
            x2 = coords_recharge2[k1][0]
            y2 = coords_recharge2[k1][1]
            x2_next = coords_recharge2[k1+1][0]
            y2_next = coords_recharge2[k1+1][1]
            dist = math.sqrt((x2_next - x2) ** 2 + (y2_next - y2) ** 2)
            time_to_recharge = path_cost_to_recharge[k1] / kvadrSpeed
            time_counter += time_to_recharge
            time.append(time_counter)
            time.append(time_counter)
            speedUGV = dist / time_to_recharge
            UGV_velocity.append(speedUGV)
            UGV_velocity.append(speedUGV)
            dx2 = x2_next - x2
            dy2 = y2_next - y2
            rads2 = math.atan2(dy2, dx2)
        if math.hypot(x2 - x2_next, y2 - y2_next) >= speedUGV:
            x2 += sm_x1
            y2 += sm_y1
            pygame.draw.circle(dis, BLUE, (x2,y2), 3)
            sm_x1 = speedUGV * math.cos(rads2)
            sm_y1 = speedUGV * math.sin(rads2)
            pygame.display.update()
        else:
            bol2 = False
        if not bol2:
            if k1 <= len(coords_recharge2) - 2:
                k1 += 1
                time_counter += recgarge_time
                time.append(time_counter)
                time.append(time_counter)
                UGV_velocity.append(0)
                UGV_velocity.append(0)

TIME = PATH_COST/kvadrSpeed
print(TIME)
UGV_velocity.append(0)
print(UGV_velocity)
print(area.square())
print(PATH_COST)
plt.plot( time, UGV_velocity)
plt.xlabel('Time')
plt.ylabel('UGV velocity')
plt.show()












