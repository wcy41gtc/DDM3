import sys

class Element:

    def __init__(self, ind: int, coords: tuple, orient: tuple, disp: tuple, stress: tuple):
        self.ind = ind  # index
        if len(coords) != 3:
            sys.exit(
                'Error! Coordinates vector must contain exactly three components.')
        else:
            self.x = coords[0]
            self.y = coords[1]
            self.z = coords[2]
            self.coords = coords  # global coordinates
        if len(orient) != 2:
            sys.exit(
                'Error! Orientation vecotr must contain exactly two components.')
        else:
            self.strike = orient[0]  # strike
            self.dip = orient[1]    # dip
            self.orient = orient    # orientation vector
        if len(disp) != 3:
            sys.exit(
                'Error! Displacement vector must contain exactly three components.')
        else:
            self.dsl = disp[0]  # shear displacement (strike-slip)
            self.dsh = disp[1]  # shear displacement (dip-slip)
            self.dnn = disp[2]  # normal displacement (opening)
            self.disp = disp    # displacemnt vector
        if len(stress) != 3:
            sys.exit('Error! Stress vector must contain exactly three components.')
        else:
            self.Ssl = stress[0]  # shear stress (strike-slip)
            self.Ssh = stress[1]  # shear stress (dip-slip)
            self.Snn = stress[2]  # normal stress (opening)
            self.stress = stress    # stress vector

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
