import sys

class Channel:

    def __init__(self, ind: int, coords: tuple, gauge_len: float):
        self.ind = ind  # index
        if len(coords) != 3:
            sys.exit(
                'Error! Coordinates vector must contain exactly three components.')
        else:
            self.x = coords[0]
            self.y = coords[1]
            self.z = coords[2]
            self.coords = coords  # global coordinates
        self.gauge_len = gauge_len
        self.Sxx = []
        self.Syy = []
        self.Szz = []
        self.Sxy = []
        self.Sxz = []
        self.Syz = []
        self.Uxx = []
        self.Uyy = []
        self.Uzz = []

    def __str__(self):
        return 'Channel id: {}\n'.format(self.ind) +\
               'Coords: {}\n'.format(self.coords) +\
               'Gauge Length: {}\n'.format(self.gauge_len)

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
    def gauge_len(self):
        return (self.gauge_len)

    @property
    def Sxx(self):
        return (self.Sxx)

    @Sxx.setter
    def Sxx(self, Sxx: float):
        self.Sxx.append(Sxx)

    @property
    def Syy(self):
        return (self.Syy)

    @Syy.setter
    def Syy(self, Syy: float):
        self.Syy.append(Syy)

    @property
    def Szz(self):
        return (self.Szz)

    @Szz.setter
    def Szz(self, Szz: float):
        self.Szz.append(Szz)
