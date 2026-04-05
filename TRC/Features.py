''''
Dealing with the broken line segments inside an archived polygon, or just a 
closure polygon.
┍━━━━━┑
┃  ┃  ┃           
┃  ┃  ┣━━━━━━━━━━━━┑
┃  ┃  ┃ ━━━━┳━━━   ┃
┃  ┣━              ┃
┕━━━━━┻━━━━━━━━━━━━┛

step.1, get all contours set (CS) that lines are not boundary
    s.11 get all skeletons and cooresponding cross points and end points
    s.22 RULE-1 ()

'''



from typing import List, Dict, Optional, Any, Tuple, Union
import numpy as np
import torch

from .algorithm import *




class Point:
    
    def __init__(self, coordinates:Union[np.ndarray, list]) -> None:
        
        self.coordinate = np.array(coordinates)
        self.x, self.y = self.coordinate
        
    
    def to(self, dst:Union['Point', 'Segment']) -> float:
        if isinstance(dst, Point):
            distance = np.linalg.norm(self - dst)
        elif isinstance(dst, Segment):
            distance = Point2Line(Point.coordinates, *dst.vertices)
        
        return distance


    def __sub__(self, other_node):
        '''override the subset function'''
        return self.coordinate - other_node.coordinate

    def move(self, distance, angle):
        '''
        the angle here -> [0, 360]
        '''
        d_x = distance * np.cos(angle)
        d_y = distance * np.sin(angle)

        self.coordinate = np.array([self.x + d_x, self.y + d_y])
        
        self.x, self.y = self.coordinate


class Segment:
    ...

    def __init__(self, vertices:List[Point]) -> None:
        assert len(vertices) == 2
        
        self.vertices = []

        for vertex in vertices:
            self.vertices.append(vertex)

        self._vec = vertices[0] - vertices[1]
        self.theta = np.arctan(self._vec[1]/  self._vec[0])

        self.line_length = np.linalg.norm(self._vec)
            
        # self.point1, self.point2 = self.vertices

    def extend_bothway(self, length:Any):
        
        self.vertices[0] = Point(extend_line(point_end=self.vertices[0], point_start=self.vertices[1], line_length=self.line_length, extend_length = float(length)))
        self.vertices[1] = Point(extend_line(point_end=self.vertices[1], point_start=self.vertices[0], line_length=self.line_length, extend_length = float(length)))
        self.line_length += 2* length
    
    def intersected(self, other: 'Segment') -> bool:
        Eps = 1e-9

        A,B = self.vertices
        C,D = other

        AC_v = C.coordinate - A.coordinate
        AD_v = D.coordinate - A.coordinate
        BC_v = C.coordinate - B.coordinate
        BD_v = D.coordinate - B.coordinate
        CA_v = A.coordinate - C.coordinate
        DA_v = A.coordinate - D.coordinate
        CB_v = B.coordinate - C.coordinate
        DB_v = B.coordinate - D.coordinate
        
        return outer_product(AC_v, AD_v) * outer_product(BC_v, BD_v) <= Eps and \
            outer_product(CA_v, CB_v) * outer_product(DA_v, DB_v) <= Eps

    def angle(self, dst:'Segment') -> float:
        '''
        get the angle [0, 90] of this and dst, 0 means parallel
        '''
        sintheta = np.cross(self._vec, dst._vec) / (self.line_length * dst.line_length)
        theta = np.abs(np.arcsin(sintheta)) * 180 / np.pi
        
        return np.min([theta, 180-theta])


    def translation(self, distance):
        ...
        # self.new_p1 = 

    

class Line:
    '''
    A identical class aims to link the individual code and spatial coordinates
    
    '''
    def __init__(self, line_code, vertices:Union[list, List[torch.Tensor]]=None, linked_nodes=None, prepared=True) -> None:
        
        self.vertices = vertices

        self.linked_nodes = [] if linked_nodes is None else linked_nodes
        self.n_linked_nodes = len(self.linked_nodes)
        
        self.linked_lines = []
        self.n_linked_lines = len(self.linked_lines)

        self.line_code = line_code
        self.is_dangling = False
        
        self.sub_segments= []
        self.prepared = prepared
        self.is_dual = False
        self.expend_width = 0
        self.length = 0
        self.line_width = 0 #mean
        self.min_line_width = 0

        self.touches = []
    
    def set_dual(self, width, is_dual=True):
        self.is_dual = is_dual
        self.expend_width = int(width)
    
    def set_length(self, length):
        self.length = length

    def set_linked_nodes(self, *node_ids):
        self.linked_nodes += node_ids
        self.n_linked_nodes = len(self.linked_nodes)

    def set_line_width(self, width, min=0):
        self.line_width = width
        self.min_line_width = min

    def set_linked_lines(self, nodes):
        for nodeid in self.linked_nodes:
            self.linked_lines += nodes[nodeid].linked_lines
        self.linked_lines = list(set(self.linked_lines))
        self.linked_lines.remove(self.line_code)
        self.n_linked_lines = len(self.linked_lines)

    def set_dangling(self):
        self.is_dangling = True
    
    def cancel_dangling(self):
        self.is_dangling = False
        
    def set_touched(self):
        self.touches.append(True)
        

class Node():
    def __init__(self, point_code, coordinate=None, linked_lines:list=None, is_dual=False) -> None:
        self.point_code = point_code
        self.coordinate = coordinate
        self.linked_lines = []
        self.is_dual = is_dual
        self.expend_width = 0
        self.linked_nodes = []
        self.n_linked_nodes = 0
    
    
    def set_coordinate(self, coordinate):
        self.coordinate = coordinate
    
    def add_linked_line(self, *line_code):
        self.linked_lines += line_code
        
    
    def set_dual(self, width:float, is_dual=True):
        self.is_dual = is_dual
        self.expend_width = int(width)
    







def outer_product(v1, v2):  
    return v1.x*v2.y - v2.x*v1.y


