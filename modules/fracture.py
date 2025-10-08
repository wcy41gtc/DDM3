import sys
from element import *


class Fracture:

    def __init__(self, ind: int, size: tuple, grid_size: tuple, center: tuple, respara: dict):
        self.ind = ind
        self.elements = []
        if len(size) != 2:
            sys.exit('Error! The size vector must contain two components')
        else:
            self.tot_l = size[0]
            self.tot_h = size[1]
            self.size = size
        if len(grid_size) != 2:
            sys.exit('Error! The grid size vector must contain two components')
        else:
            self.dl = grid_size[0]
            self.dh = grid_size[1]
            self.grid_size = grid_size
        if len(center) != 3:
            sys.exit('Error! The center vector must contain three components')
        else:
            self.c_x = center[0]
            self.c_y = center[1]
            self.c_z = center[2]
            self.center = center
        self.PoissonRatio = respara['PoissonRatio']
        self.ShearModulus = respara['ShearModulus']

    # discretize Fracture to a list of Elements
    def discretize(self):
        n = 1  # initialize id counter
        for i in range(1, int(self.tot_l / self.dl) + 1):
            for j in range(1, int(self.tot_h / self.dh) + 1):
                x = self.c_x - self.tot_l / 2.0 + (self.dl) / 2 * (2 * i - 1)
                y = self.c_y
                z = self.c_z - self.tot_h / 2 + (self.dh) / 2 * (2 * j - 1)
                self.elements.append(
                    Element(n, (x, y, z), (dl, dh), (0, 90),
                            (dsl, dsh, dnn), (Ssl, Ssh, Snn))
                )  # only create 0 strike 90 dip fractures for now
                n += 1
        print('Square fracture discretized:\ncenter: {},\n# of Elements: {}\n'.format(
            self.center, len(self.elements)))
