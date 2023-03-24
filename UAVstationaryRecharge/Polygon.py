import math

import pygame
from pygame.draw_py import Point
from shapely import Polygon
import numpy as np
import Utils as ut

class Polygon2:

    def __init__(self,coords,dis):
        self.coords = coords
        self.dis = dis
        self.poly = Polygon(coords)

    def pointInPolygon(self,x, y):
        c = 0
        for i in range(len(self.coords)):
            if (((self.coords[i][1] <= y and y < self.coords[i-1][1]) or (self.coords[i-1][1] <= y and y < self.coords[i][1])) and
                    (x > (self.coords[i-1][0]- self.coords[i][0]) * (y - self.coords[i][1]) / (self.coords[i-1][1] - self.coords[i][1]) + self.coords[i][0])): c = 1 - c
        return c

    def drawPolygon(self,color):
        for i in range(len(self.coords) - 1):
            pygame.draw.line(self.dis, color, self.coords[i], self.coords[i + 1], 8)
            if i == len(self.coords) - 2:
                pygame.draw.line(self.dis, color, self.coords[i + 1], self.coords[0], 8)

    def drawRectangularGrid(self,wx,wy):
        cooordRect = []
        xMax = self.coords[0][0]
        ixMax = 0
        yMax = self.coords[0][1]
        iyMax = 0
        xMin = self.coords[0][0]
        ixMin = 0
        yMin = self.coords[0][1]
        iyMin = 0
        for i in range(len(self.coords)):
            if xMax < self.coords[i][0]:
                xMax = self.coords[i][0]
                iXMax = i
        cooordRect.append((xMax, self.coords[iXMax][1]))
        for i in range(len(self.coords)):
            if yMax < self.coords[i][1]:
                yMax = self.coords[i][1]
                iyMax = i
        cooordRect.append((self.coords[iyMax][0], yMax))
        for i in range(len(self.coords)):
            if xMin > self.coords[i][0]:
                xMin = self.coords[i][0]
                ixMin = i
        cooordRect.append((xMin, self.coords[ixMin][1]))
        for i in range(len(self.coords)):
            if yMin > self.coords[i][1]:
                yMin = self.coords[i][1]
                iyMin = i
        cooordRect.append((self.coords[iyMin][0], yMin))

        coordR = [xMin + wx/2, yMin + wy/2]
        coordInArea = []
        coordInAreaCenter = []
        coordAll = []
        line = 0
        i = 0
        maskaCoord = []
        while coordR[1] < yMax:
            coordAll.append(coordR[0])
            coordGrid = [(coordR[0] - wx/2, coordR[1]-wy/2),(coordR[0] - wx/2, coordR[1]+wy/2),
                         (coordR[0] + wx/2 , coordR[1]+wy/2),(coordR[0] + wx/2, coordR[1]-wy/2)]
            gridOne = Polygon(coordGrid)
            if self.poly.intersects(gridOne) == 1:
                maskaCoord.append(0)
                coordInArea.append(Point(coordR[0] - wx / 2, coordR[1] - wy / 2))
                coordInAreaCenter.append(Point(coordR[0], coordR[1]))
                pygame.draw.rect(self.dis, (0, 255, 0), (coordInArea[i].x, coordInArea[i].y, wx, wy), 2)
                i += 1
            else:
                maskaCoord.append(1)
                pygame.draw.rect(self.dis, (21, 47, 188), (coordR[0] - wx / 2, coordR[1] - wy / 2, wx, wy), 1)
            coordR[0] += wx

            if coordR[0] > xMax:
                line += 1
                coordR = [xMin + wx/2,yMin + (line + 1/2)*wy]
        return(coordAll,line,yMin + wy/2,maskaCoord)

    def square(self):
        return self.poly.area

    def findOptimalTriangle(self, area, dis):
        highestVertexCoords, index = ut.findHighestVertexCoord(self.coords)
        opt_triangle_coords = []
        opt_triangle_coords.append(self.coords[ut.getNextLeftIndex(self.coords, index)])
        indexToRightSide = ut.getNextRightIndex(self.coords, index)
        angleCurrent = 0
        angleBorder = ut.getAngleBy3Points(self.coords[indexToRightSide],
                                           highestVertexCoords, self.coords[ut.getNextLeftIndex(self.coords, index)])
        angleCurrent = 0
        status = True
        indexNearVertex = index
        coordsBeginStep2 = self.coords[ut.getNextLeftIndex(self.coords, index)]
        while angleBorder >= angleCurrent:
            if status == True:
                indexNearVertex = ut.getNextLeftIndex(self.coords, indexNearVertex)
                indexNearNearVertex = ut.getNextLeftIndex(self.coords, indexNearVertex)
                length_x = self.coords[indexNearVertex][0] - self.coords[indexNearNearVertex][0]
                length_y = self.coords[indexNearVertex][1] - self.coords[indexNearNearVertex][1]
                step_x = -length_x/100
                step_y = -length_y/100
                coordsBeginStep = self.coords[indexNearVertex]
                coordsBeginStep = list(coordsBeginStep)
                angleSegment = ut.getAngleBy3Points(self.coords[indexNearNearVertex],
                                                    highestVertexCoords,
                                                    self.coords[ut.getNextLeftIndex(self.coords, index)])
            while angleSegment > angleCurrent:
                coordsBeginStep = list(coordsBeginStep)
                coordsBeginStep[0] += step_x
                coordsBeginStep[1] += step_y
                coordsBeginStep = tuple(coordsBeginStep)
                angleCurrent = ut.getAngleBy3Points(coordsBeginStep, highestVertexCoords,
                                                    self.coords[ut.getNextLeftIndex(self.coords, index)])
                polyCurrent = Polygon2([highestVertexCoords, coordsBeginStep2, coordsBeginStep], dis)
                if polyCurrent.square() >= area:
                    status = False
                    coordsBeginStep2 = coordsBeginStep
                    opt_triangle_coords.append(coordsBeginStep)
                    break
            if angleSegment <= angleCurrent:
                status = True
        opt_triangle_coords.append(self.coords[ut.getNextRightIndex(self.coords, index)])
        return opt_triangle_coords


    def create_grid_for_triangle2(tr_coords1, tr_coords2, highest_vertex, wx, wy, dis, color):

        angle_OY = ut.find_angle_OY(ut.find_k(tr_coords1, highest_vertex))
        highest_vertex_n = ut.coord_transform_to_NewCSx(angle_OY, highest_vertex)
        tr_coords1_n = ut.coord_transform_to_NewCSx(angle_OY, tr_coords1)
        tr_coords2_n = ut.coord_transform_to_NewCSx(angle_OY, tr_coords2)
        cooordRect = []
        cooordRect.append(list(highest_vertex_n))
        if tr_coords1_n[1] >= tr_coords2_n[1]:
            cooordRect.append(list(tr_coords1_n))
            cooordRect.append([tr_coords2_n[0], tr_coords1_n[1]])
        else:
            cooordRect.append([tr_coords1_n[0], tr_coords2_n[1]])
            cooordRect.append(list(tr_coords2_n))
        cooordRect.append([tr_coords2_n[0], highest_vertex_n[1]])
        tri_poly = Polygon([highest_vertex_n, tr_coords1_n, tr_coords2_n])
        coordR = [cooordRect[0][0] + wx / 2, cooordRect[0][1] + wy / 2]
        coordInArea = []
        coordInAreaCenter = []
        coordAll = []
        line = 0
        i = 0
        yMax = cooordRect[1][1]
        xMax = cooordRect[3][0]
        xMin = cooordRect[0][0]
        yMin = cooordRect[0][1]
        maskaCoord = []
        yForStr = []

        while coordR[1] < yMax:
            coord_in_oldCS = ut.coord_transform_to_OldCSx(angle_OY, coordR)
            coordAll.append(coord_in_oldCS[0])
            yForStr.append(coord_in_oldCS[1])
            coordGrid = [(coordR[0] - wx / 2, coordR[1] - wy / 2), (coordR[0] - wx / 2, coordR[1] + wy / 2),
                         (coordR[0] + wx / 2, coordR[1] + wy / 2), (coordR[0] + wx / 2, coordR[1] - wy / 2)]
            gridOne = Polygon(coordGrid)
            if tri_poly.intersects(gridOne) == 1:
                maskaCoord.append(0)
                tup = [coordR[0] - wx / 2, coordR[1] - wy / 2]
                x_in_area = ut.coord_transform_to_OldCS_x(angle_OY, tup)
                y_in_area = ut.coord_transform_to_OldCS_y(angle_OY, tup)
                tup2 = [coordR[0], coordR[1]]
                x_in_acenter = ut.coord_transform_to_OldCSx(angle_OY, tup2)
                y_in_acenter = ut.coord_transform_to_OldCSx(angle_OY, tup2)
                # coordInArea.append(Point(x_in_area, y_in_area))
                # coordInAreaCenter.append(Point(x_in_acenter, y_in_acenter))
                # pygame.draw.rect(dis, (0, 255, 0), (coordInArea[i].x, coordInArea[i].y, wx, wy), 2)
                for i in range(len(coordGrid)):
                    coordGrid[i] = ut.coord_transform_to_OldCSx(angle_OY, coordGrid[i])
                for i in range(len(coordGrid) - 1):
                    pygame.draw.line(dis, color, coordGrid[i], coordGrid[i + 1], 1)
                pygame.draw.line(dis, color, coordGrid[3], coordGrid[0], 1)
                i += 1
            else:
                maskaCoord.append(1)
            coordR[0] += wx
            if coordR[0] > xMax:
                # yForStr.append(ut.coord_transform_to_OldCS_y(angle_OY, [coordR[0] - wx, coordR[1]]))
                line += 1
                coordR = [xMin + wx / 2, yMin + (line + 1 / 2) * wy]
        maskaCoord = np.array(maskaCoord)
        maskaCoord = np.reshape(maskaCoord, (line, int(maskaCoord.shape[0] / line)))
        points = np.array(coordAll)
        points = np.reshape(points, (line, int(points.shape[0] / line)))
        yForStr = np.array(yForStr)
        yForStr = np.reshape(yForStr, (line, int(yForStr.shape[0] / line)))

        return (points, line, yForStr, maskaCoord)





















            































