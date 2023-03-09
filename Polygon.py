import pygame
from GeneticAlg import *
from shapely import Polygon
import numpy as np

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
            pygame.draw.line(self.dis, color, self.coords[i], self.coords[i + 1], 5)
            if i == len(self.coords) - 2:
                pygame.draw.line(self.dis, color, self.coords[i + 1], self.coords[0], 5)

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
        cooordRect.append((xMax,self.coords[iXMax][1]))
        for i in range(len(self.coords)):
            if yMax < self.coords[i][1]:
                yMax = self.coords[i][1]
                iyMax = i
        cooordRect.append((self.coords[iyMax][0],yMax))
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
        coordR = [xMin + wx/2,yMin + wy/2]
        coordInArea = []
        coordInAreaCenter = []
        coordAll = []
        line = 0
        i=0
        maskaCoord = []
        while coordR[1] < yMax:
            coordAll.append(coordR[0])
            coordGrid = [(coordR[0] - wx/2, coordR[1]-wy/2),(coordR[0] - wx/2, coordR[1]+wy/2),(coordR[0] + wx/2 , coordR[1]+wy/2),(coordR[0] + wx/2, coordR[1]-wy/2)]
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


















