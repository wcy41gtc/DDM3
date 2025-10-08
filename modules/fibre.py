import sys
from channel import *


class Fibre:

    num_of_fibres = 0

    def __init__(self, ind: int, start_coords: tuple, end_coords: tuple, nchn: int):
        self.ind = ind  # index
        if len(start_coords) != 3:
            sys.exit(
                'Error! Coordinates vector must contain exactly three components.')
        else:
            self.sx = start_coords[0]
            self.sy = start_coords[1]
            self.sz = start_coords[2]
            self.start_coords = start_coords  # global coordinates
        if len(end_coords) != 3:
            sys.exit(
                'Error! Coordinates vector must contain exactly three components.')
        else:
            self.ex = end_coords[0]
            self.ey = end_coords[1]
            self.ez = end_coords[2]
            self.end_coords = end_coords  # global coordinates
        if nchn == 0:
            sys.exit(
                'Error! Number of Channels cannot be zero.')
        else:
            self.nchn = nchn  # number of channels

# need work
        self.Sxx

    def __str__(self):
        return 'Element id: {}\n'.format(self.ind) +\
               'Coords: {}\n'.format(self.coords) +\
               'Orientation: {}\n'.format(self.orient) +\
               'Displacements: {}\n'.format(self.disp) +\
               'Stresses: {}\n'.format(self.stress)

    @property
    def x(self):
        return self.x

    @property
    def y(self):
        return self.y

    @property
    def z(self):
        return self.z

    @property
    def coords(self):
        return (self.x, self.y, self.z)

    @property
    def orient(self):
        return (self.strike, self.dip)

    @property
    def strike(self):
        return self.strike

    @property
    def dip(self):
        return self.dip

    @property
    def disp(self):
        return (self.dsl, self.dsh, self.dnn)

    @disp.setter
    def disp(self, disp: tuple):
        if len(disp) != 3:
            sys.exit(
                'Error! Displacement vector must contain exactly three components.')
        else:
            self.dsl = disp[0]
            self.dsh = disp[1]
            self.dnn = disp[2]
            self.disp = disp

    @property
    def dsl(self):
        return self.dsl

    @dsl.setter
    def dsl(self, dsl):
        self.dsl = dsl

    @property
    def dsh(self):
        return self.dsh

    @dsh.setter
    def dsh(self, dsh):
        self.dsh = dsh

    @property
    def dnn(self):
        return self.dnn

    @dnn.setter
    def dnn(self, dnn):
        self.dnn = dnn

    @property
    def stress(self):
        return (self.Ssl, self.Ssh, self.Snn)

    @stress.setter
    def stress(self, stress: tuple):
        if len(stress) != 3:
            sys.exit('Error! Stress vector must contain exactly three components.')
        else:
            self.Ssl = stress[0]
            self.Ssh = stress[1]
            self.Snn = stress[2]
            self.stress = stress

    @property
    def Ssl(self):
        return self.Ssl

    @dsl.setter
    def Ssl(self, Ssl):
        self.Ssl = Ssl

    @property
    def Ssh(self):
        return self.Ssh

    @dsh.setter
    def Ssh(self, dsh):
        self.Ssh = Ssh

    @property
    def Snn(self):
        return self.Snn

    @Snn.setter
    def Snn(self, dnn):
        self.Snn = Snn
