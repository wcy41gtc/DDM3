import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import median_filter

def make_dde(n,x,y,z,dl,dh,sk,dp,yw,dsl,dsh,dnn,Ssl,Ssh,Snn):
    '''
    create a dictionary container for the displacement discontinuity element
    n: id,
    x, y, z: global coordinates
    s: Strike
    dsl, Ssl: shear displacement and stress (strike-slip)
    dsh, Ssh: shear displacement and stress (dip-slip)
    dnn, Snn: normal displacement and stress (opening)
    
    '''
    return {'id':n,'X':x,'Y':y,'Z':z,'dl':dl,'dh':dh,'strike':sk,'dip':dp,'yaw':yw,
            'dsl':dsl,'dsh':dsh,'dnn':dnn,'Ssl':Ssl,'Ssh':Ssh,'Snn':Snn}

def make_fracture(tot_l,tot_h,dl,dh,c_x,c_y,c_z,sk,dp,yw,dsl,dsh,dnn,Ssl,Ssh,Snn):
    '''
    make a square fracture with 0 degree strike
    tot_l: total length of the fracture
    tot_h: total height of the fracture
    dl: length increment
    dh: height increment
    c_x, c_y, c_z: global coordinates of the fracture's center
    dsl, Ssl: shear displacement and stress (strike-slip)
    dsh, Ssh: shear displacement and stress (dip-slip)
    dnn, Snn: normal displacement and stress (opening)
    '''
    n = 1 #initialize id counter
    fracture = [] #initialize fracture, i.e., an empty list
    cos_SK = np.cos(np.deg2rad(sk))
    sin_SK = np.sin(np.deg2rad(sk))
    
    cos_DP = np.cos(np.deg2rad(dp))
    sin_DP = np.sin(np.deg2rad(dp))
    
    cos_YW = np.cos(np.deg2rad(yw))
    sin_YW = np.sin(np.deg2rad(yw))
    
    SK = np.array([[cos_SK,sin_SK,0.0],[-sin_SK,cos_SK,0.0],[0.0,0.0,1.0]])
    DP = np.array([[1.0,0.0,0.0],[0.0,cos_DP,sin_DP],[0.0,-sin_DP,cos_DP]])
    YW = np.array([[cos_YW,sin_YW,0.0],[-sin_YW,cos_YW,0.0],[0.0,0.0,1.0]])
    
    AA = np.matmul(np.matmul(SK,DP),YW)
    for i in range(1,round(tot_l/dl)):
        for j in range(1,round(tot_h/dh)):
            x = c_x-tot_l/2+dl/2*(2*i-1)
            y = c_y
            z = c_z-tot_h/2+dh/2*(2*j-1)
            lo = np.array([[x-c_x],[y-c_y],[z-c_z]]) #local coordinates
            # coordinates trasform
            gl = np.matmul(AA,lo)
            fracture.append(make_dde(n,gl[0][0]+c_x,gl[1][0]+c_y,gl[2][0]+c_z,dl,dh,sk,dp,yw,dsl,dsh,dnn,Ssl,Ssh,Snn))
            n += 1
#     print(len(fracture))
    return fracture

def make_fracture_ellipse(tot_l,tot_h,dl,dh,c_x,c_y,c_z,sk,dp,yw,dsl,dsh,dnn,Ssl,Ssh,Snn):
    '''
    make a elliptical fracture with 0 degree strike
    tot_l: total length of the fracture
    tot_h: total height of the fracture
    dl: length increment
    dh: height increment
    c_x, c_y, c_z: global coordinates of the fracture's center
    dsl, Ssl: shear displacement and stress (strike-slip)
    dsh, Ssh: shear displacement and stress (dip-slip)
    dnn, Snn: normal displacement and stress (opening)
    '''
    n = 1 #initialize id counter
    fracture = [] #initialize fracture, i.e., an empty list
    cos_SK = np.cos(np.deg2rad(sk))
    sin_SK = np.sin(np.deg2rad(sk))
    
    cos_DP = np.cos(np.deg2rad(dp))
    sin_DP = np.sin(np.deg2rad(dp))
    
    cos_YW = np.cos(np.deg2rad(yw))
    sin_YW = np.sin(np.deg2rad(yw))
    
    SK = np.array([[cos_SK,sin_SK,0.0],[-sin_SK,cos_SK,0.0],[0.0,0.0,1.0]])
    DP = np.array([[1.0,0.0,0.0],[0.0,cos_DP,sin_DP],[0.0,-sin_DP,cos_DP]])
    YW = np.array([[cos_YW,sin_YW,0.0],[-sin_YW,cos_YW,0.0],[0.0,0.0,1.0]])
    
    AA = np.matmul(np.matmul(SK,DP),YW)
    for i in range(1,int(tot_l/dl)):
        for j in range(1,int(tot_h/dh)):
            x = c_x-tot_l/2+dl/2*(2*i-1)
            y = c_y
            z = c_z-tot_h/2+dh/2*(2*j-1)
            tester = (x-c_x)**2/(tot_l/2)**2+(z-c_z)**2/(tot_h/2)**2
            if tester <= 1:
                lo = np.array([[x-c_x],[y-c_y],[z-c_z]]) #local coordinates
                # coordinates trasform
                gl = np.matmul(AA,lo)
                fracture.append(make_dde(n,gl[0][0]+c_x,gl[1][0]+c_y,gl[2][0]+c_z,dl,dh,sk,dp,yw,dsl,dsh,dnn,Ssl,Ssh,Snn))
                n += 1
    return fracture

def make_fibre(start:tuple=(0,0,0),end:tuple=(0,0,0),n_chn:int=0):
    '''
    create a fibre with global coordinates
    start: starting coordinates (X,Y,Z)
    end: ending coordinates (X,Y,Z)
    n_chn: number of channels
    '''
    fibre = []
    chn = {}
    for i in range(1,n_chn+1):
        chn = {'id': i,
               'X': start[0]+(2*i-1)*(end[0]-start[0])/(2*n_chn),
               'Y': start[1]+(2*i-1)*(end[1]-start[1])/(2*n_chn),
               'Z': start[2]+(2*i-1)*(end[2]-start[2])/(2*n_chn),
               'dl': np.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2+(end[2]-start[2])**2)/n_chn,
               'SXX': [],
               'SYY': [],
               'SZZ': [],
               'SXY': [],
               'SXZ': [],
               'SYZ': [],
               'EXX': [],
               'EYY': [],
               'EZZ': [],
               'UXX': [],
               'UYY': [],
               'UZZ': []}
        fibre.append(chn)
    return fibre


def make_plane(center:tuple,size:tuple,node_size:tuple,orientation):
    '''
    create a plane with global coordinates
    center: center coordinates of the plane (X,Y,Z)
    tot_l: ending coordinates (X,Y,Z)
    loc: number of channels
    '''
    plane = []
    node = {}
    n_nodes_l = int(size[0]/node_size[0])
    n_nodes_w = int(size[1]/node_size[1])
    counter = 1
    if orientation == 'XZ':
        for i in range(1,n_nodes_l+1):
            for j in range(1,n_nodes_w+1):
                node = {'id': counter,
                        'row':i-1,
                        'col':j-1,
                        'X': center[0]-size[0]/2+node_size[0]/2*(2*i-1),
                        'Y': center[1],
                        'Z': center[2]-size[1]/2+node_size[1]/2*(2*j-1)}
                counter += 1
                plane.append(node)
    elif orientation == 'XY':
        for i in range(1,n_nodes_l+1):
            for j in range(1,n_nodes_w+1):
                node = {'id': counter,
                        'row':i-1,
                        'col':j-1,
                        'X': center[0]-size[0]/2+node_size[0]/2*(2*i-1),
                        'Y': center[1]-size[1]/2+node_size[1]/2*(2*j-1),
                        'Z': center[2]}
                counter += 1
                plane.append(node)
    elif orientation == 'YZ':
        for i in range(1,n_nodes_l+1):
            for j in range(1,n_nodes_w+1):
                node = {'id': counter,
                        'row':i-1,
                        'col':j-1,
                        'X': center[0],
                        'Y': center[1]-size[0]/2+node_size[0]/2*(2*i-1),
                        'Z': center[2]-size[1]/2+node_size[1]/2*(2*j-1)}
                counter += 1
                plane.append(node)
    for node in plane:
        node['center'] = center
        node['size'] = size
        node['node_size'] = node_size
        node['orientation'] = orientation
        node['SXX'] = []
        node['SYY'] = []
        node['SZZ'] = []
        node['SXY'] = []
        node['SXZ'] = []
        node['SYZ'] = []
        node['EXX'] = []
        node['EYY'] = []
        node['EZZ'] = []
        node['UXX'] = []
        node['UYY'] = []
        node['UZZ'] = []
    return plane

def clear_fibre_stress_disp(fibre):
    '''
    clear the stresses and displacements stored in fibre
    '''
    for chn in fibre:
        chn['SXX'] = []
        chn['SYY'] = []
        chn['SZZ'] = []
        chn['SXY'] = []
        chn['SXZ'] = []
        chn['SYZ'] = []
        chn['EXX'] = []
        chn['EYY'] = []
        chn['EZZ'] = []
        chn['UXX'] = []
        chn['UYY'] = []
        chn['UZZ'] = []
    return fibre

def cal_dd(fractures:list,respara:dict):
    '''
    calculate the displacement discontinuity based on initial stress conditions
    fracture: a list of displacement discontinuity elements (dde)
    SXX, SYY, SZZ: principal stresses along the global axes
    '''
    # ignore runtime warning
    # import warnings
    # warnings.filterwarnings("ignore", category=RuntimeWarning) 
    #generate linear system matrix
    for fracture in fractures:
        coef_slsl = np.zeros((len(fracture),len(fracture)))
        coef_slsh = np.zeros((len(fracture),len(fracture)))
        coef_slnn = np.zeros((len(fracture),len(fracture)))
        coef_shsl = np.zeros((len(fracture),len(fracture)))
        coef_shsh = np.zeros((len(fracture),len(fracture)))
        coef_shnn = np.zeros((len(fracture),len(fracture)))
        coef_nnsl = np.zeros((len(fracture),len(fracture)))
        coef_nnsh = np.zeros((len(fracture),len(fracture)))
        coef_nnnn = np.zeros((len(fracture),len(fracture)))
        Ssl = np.zeros((len(fracture),1))
        Ssh = np.zeros((len(fracture),1))
        Snn = np.zeros((len(fracture),1))
        for i, r_dde in enumerate(fracture): #receiver dde
            Ssl[i] = r_dde['Ssl']
            Ssh[i] = r_dde['Ssh']
            Snn[i] = r_dde['Snn']
            for j, dde in enumerate(fracture):
                gamma = r_dde['strike']-dde['strike'] #difference in strike for local coordinates
                cos_gamma = np.cos(np.deg2rad(gamma))
                cos_2gamma = np.cos(np.deg2rad(2*gamma))
                sin_gamma = np.sin(np.deg2rad(gamma))
                sin_2gamma = np.sin(np.deg2rad(2*gamma))

#                 cos_beta = np.cos(np.deg2rad(r_dde['strike']))
#                 cos_2beta = np.cos(np.deg2rad(2*r_dde['strike']))
#                 sin_beta = np.sin(np.deg2rad(r_dde['strike']))
#                 sin_2beta = np.sin(np.deg2rad(2*r_dde['strike']))
                x1 = (r_dde['X']-dde['X'])*cos_gamma+(r_dde['Y']-dde['Y'])*sin_gamma
                x2 = r_dde['Z']-dde['Z']
                x3 = -(r_dde['X']-dde['X'])*sin_gamma+(r_dde['Y']-dde['Y'])*cos_gamma
                a = dde['dl']/2.0
                b = dde['dh']/2.0
                #calculate coefficient
                Cr = respara['ShearModulus']/(4.0*np.pi*(1.0-respara['PoissonRatio']))
                r1 = np.sqrt((x1-a)**2+(x2-b)**2+x3**2)
                r2 = np.sqrt((x1-a)**2+(x2+b)**2+x3**2)
                r3 = np.sqrt((x1+a)**2+(x2-b)**2+x3**2)
                r4 = np.sqrt((x1+a)**2+(x2+b)**2+x3**2)
                #I,1
                J1 = np.log(r1+x2-b)-np.log(r2+x2+b)-np.log(r3+x2-b)+np.log(r4+x2+b)
                #I,2
                J2 = np.log(r1+x1-a)-np.log(r2+x1-a)-np.log(r3+x1+a)+np.log(r4+x1+a)
                #I,3
                J3 = -np.arctan((x1-a)*(x2-b)/(x3*r1))+np.arctan((x1-a)*(x2+b)/(x3*r2))+\
                     np.arctan((x1+a)*(x2-b)/(x3*r3))-np.arctan((x1+a)*(x2+b)/(x3*r4))
                #I,11
                J4 = (x1-a)/(r1*(r1+x2-b))-(x1-a)/(r2*(r2+x2+b))-(x1+a)/(r3*(r3+x2-b))+(x1+a)/(r4*(r4+x2+b))
                #I,22
                J5 = (x2-b)/(r1*(r1+x1-a))-(x2+b)/(r2*(r2+x1-a))-(x2-b)/(r3*(r3+x1+a))+(x2+b)/(r4*(r4+x1+a))
                #I,33
                J6 = (x1-a)*(x2-b)*(x3**2+r1**2)/(r1*(x3**2+(x1-a)**2)*(x3**2+(x2-b)**2))-\
                     (x1-a)*(x2+b)*(x3**2+r2**2)/(r2*(x3**2+(x1-a)**2)*(x3**2+(x2+b)**2))-\
                     (x1+a)*(x2-b)*(x3**2+r3**2)/(r3*(x3**2+(x1+a)**2)*(x3**2+(x2-b)**2))+\
                     (x1+a)*(x2+b)*(x3**2+r4**2)/(r4*(x3**2+(x1+a)**2)*(x3**2+(x2+b)**2))
                #I,12
                J7 = 1/r1-1/r2-1/r3+1/r4
                #I,13
                J8 = x3/(r1*(r1+x2-b))-x3/(r2*(r2+x2+b))-x3/(r3*(r3+x2-b))+x3/(r4*(r4+x2+b))
                #I,23
                J9 = x3/(r1*(r1+x1-a))-x3/(r2*(r2+x1-a))-x3/(r3*(r3+x1+a))+x3/(r4*(r4+x1+a))
                #I,32
                J9_1 = x3/(r1*(r1+x1-a))-x3/(r2*(r2+x1-a))-x3/(r3*(r3+x1+a))+x3/(r4*(r4+x1+a))
                #I,111
                J10 = (-(r1+x2-b)*((x1-a)**2-r1**2)-(x1-a)**2*r1)/(r1**3*(r1+x2-b)**2)-\
                      (-(r2+x2+b)*((x1-a)**2-r2**2)-(x1-a)**2*r2)/(r2**3*(r2+x2+b)**2)-\
                      (-(r3+x2-b)*((x1+a)**2-r3**2)-(x1+a)**2*r3)/(r3**3*(r3+x2-b)**2)+\
                      (-(r4+x2+b)*((x1+a)**2-r4**2)-(x1+a)**2*r4)/(r4**3*(r1+x2+b)**2)
                #I,211
                J11 = (-(x1-a)/r1**3)-(-(x1-a)/r2**3)-(-(x1+a)/r3**3)+(-(x1+a)/r4**3)
                #I,311
                J12 = (-(x1-a)*x3*(2*r1+x2-b))/(r1**3*(r1+x2-b)**2)-\
                      (-(x1-a)*x3*(2*r2+x2+b))/(r2**3*(r2+x2+b)**2)-\
                      (-(x1+a)*x3*(2*r3+x2-b))/(r3**3*(r3+x2-b)**2)+\
                      (-(x1+a)*x3*(2*r4+x2+b))/(r4**3*(r4+x2+b)**2)
                #I,311
                J13 = (-(x2-b))/r1**3-(-(x2+b))/r2**3-(-(x2-b))/r3**3+(-(x2+b))/r4**3
                #I,122
                J14 = (-(r1+x1-a)*((x2-b)**2-r1**2)-(x2-b)**2*r1)/(r1**3*(r1+x1-a)**2)-\
                      (-(r2+x1-a)*((x2+b)**2-r2**2)-(x2+b)**2*r2)/(r2**3*(r2+x1-a)**2)-\
                      (-(r3+x1+a)*((x2-b)**2-r3**2)-(x2-b)**2*r3)/(r3**3*(r3+x1+a)**2)+\
                      (-(r4+x1+a)*((x2+b)**2-r4**2)-(x2+b)**2*r4)/(r4**3*(r4+x1+a)**2)
                #I,222
                J15 = (-(x2-b)*x3*(2*r1+x1-a))/(r1**3*(r1+x1-a)**2)-\
                      (-(x2+b)*x3*(2*r2+x1-a))/(r2**3*(r2+x1-a)**2)-\
                      (-(x2-b)*x3*(2*r3+x1+a))/(r3**3*(r3+x1+a)**2)+\
                      (-(x2+b)*x3*(2*r4+x1+a))/(r4**3*(r4+x1+a)**2)
                #I,322
                J16 = (-(r1+x2-b)*(x3**2-r1**2)-x3**2*r1)/(r1**3*(r1+x2-b)**2)-\
                      (-(r2+x2+b)*(x3**2-r2**2)-x3**2*r2)/(r2**3*(r2+x2+b)**2)-\
                      (-(r3+x2-b)*(x3**2-r3**2)-x3**2*r3)/(r3**3*(r3+x2-b)**2)+\
                      (-(r4+x2+b)*(x3**2-r4**2)-x3**2*r4)/(r4**3*(r4+x2+b)**2)
                #I,233
                J17 = (-(r1+x1-a)*(x3**2-r1**2)-x3**2*r1)/(r1**3*(r1+x1-a)**2)-\
                      (-(r2+x1-a)*(x3**2-r2**2)-x3**2*r2)/(r2**3*(r2+x1-a)**2)-\
                      (-(r3+x1+a)*(x3**2-r3**2)-x3**2*r3)/(r3**3*(r3+x1+a)**2)+\
                      (-(r4+x1+a)*(x3**2-r4**2)-x3**2*r4)/(r4**3*(r4+x1+a)**2)
                #I,333
                J18 = (-x3*(x1-a)*(x2-b))*((x3**2+(x1-a)**2)**2*(x3**2+(x2-b)**2+2*r1**2)+\
                      (x3**2+(x2-b)**2)**2*(x3**2+(x1-a)**2+2*r1**2))/(r1**3*(x3**2+(x1-a)**2)**2*(x3**2+(x2-b)**2)**2)-\
                      (-x3*(x1-a)*(x2+b))*((x3**2+(x1-a)**2)**2*(x3**2+(x2+b)**2+2*r2**2)+\
                      (x3**2+(x2+b)**2)**2*(x3**2+(x1-a)**2+2*r2**2))/(r2**3*(x3**2+(x1-a)**2)**2*(x3**2+(x2+b)**2)**2)-\
                      (-x3*(x1+a)*(x2-b))*((x3**2+(x1+a)**2)**2*(x3**2+(x2-b)**2+2*r3**2)+\
                      (x3**2+(x2-b)**2)**2*(x3**2+(x1+a)**2+2*r3**2))/(r3**3*(x3**2+(x1+a)**2)**2*(x3**2+(x2-b)**2)**2)+\
                      (-x3*(x1+a)*(x2+b))*((x3**2+(x1+a)**2)**2*(x3**2+(x2+b)**2+2*r4**2)+\
                      (x3**2+(x2+b)**2)**2*(x3**2+(x1+a)**2+2*r4**2))/(r4**3*(x3**2+(x1+a)**2)**2*(x3**2+(x2+b)**2)**2)
                #I,123
                J19 = (-x3/r1**3)-(-x3/r2**3)-(-x3/r3**3)+(-x3/r4**3)
                #coefficients
                #slsl, slsh, slnn
                coef_slsl[i,j] = Cr*(-sin_gamma*cos_gamma*(2*J8-x3*J10)+\
                                     cos_2gamma*(J6+respara['PoissonRatio']*J5-x3*J12)+sin_gamma*cos_gamma*(-x3*J16))
                coef_slsh[i,j] = Cr*(-sin_gamma*cos_gamma*(2*respara['PoissonRatio']*J9-x3*J11)+\
                                     cos_2gamma*(-respara['PoissonRatio']*J7-x3*J19)+sin_gamma*cos_gamma*(-x3*J17))
                coef_slnn[i,j] = Cr*(-sin_gamma*cos_gamma*(J6+(1-2*respara['PoissonRatio'])*J5-x3*J12)+\
                                     cos_2gamma*(-x3*J16)+sin_gamma*cos_gamma*(J6-x3*J18))
                #shsl, shsh, shnn
                coef_shsl[i,j] = Cr*(-sin_gamma*((1-respara['PoissonRatio'])*J9-x3*J11)+\
                                     cos_gamma*(-respara['PoissonRatio']*J7-x3*J19))
                coef_shsh[i,j] = Cr*(-sin_gamma*((1-respara['PoissonRatio'])*J8-x3*J13)+\
                                     cos_gamma*(J6+respara['PoissonRatio']*J4-x3*J15))
                coef_shnn[i,j] = Cr*(-sin_gamma*(-(1-2*respara['PoissonRatio'])*J7-x3*J19)+\
                                     cos_gamma*(-x3*J17))
                #nnsl, nnsh, nnnn
                coef_nnsl[i,j] = Cr*(sin_gamma**2*(2*J8-x3*J10)-\
                                     2*sin_gamma*cos_gamma*(J6+respara['PoissonRatio']*J5-x3*J12)+cos_gamma**2*(-x3*J16))
                coef_nnsh[i,j] = Cr*(sin_gamma**2*(2*respara['PoissonRatio']*J9-x3*J11)-\
                                     2*sin_gamma*cos_gamma*(-respara['PoissonRatio']*J7-x3*J19)+cos_gamma**2*(-x3*J17))
                coef_nnnn[i,j] = Cr*(sin_gamma**2*(J6+(1-2*respara['PoissonRatio'])*J5-x3*J12)-\
                                     2*sin_gamma*cos_gamma*(-x3*J16)+cos_gamma**2*(J6-x3*J18))
        #assemble coef matrix and displacement column vector
        S = np.vstack((Ssl,Ssh,Snn))
        coef = np.vstack((np.hstack((coef_slsl, coef_slsh, coef_slnn)),
                          np.hstack((coef_shsl, coef_shsh, coef_shnn)),
                          np.hstack((coef_nnsl, coef_nnsh, coef_nnnn))))
    #     if np.isnan(coef).any():
    #         print('NaN found in calculation')
        coef[np.isnan(coef)] = 0.0
        coef[coef<1e-10] = 0.0
        #solve for DD
#         DD = np.linalg.solve(coef,S)
        #print(np.linalg.cond(coef))
        DD = np.linalg.lstsq(coef,S,rcond=None)[0]
        # aign DD back to dde in fracture
        DD[DD<1e-10] = 0
        for k, r_dde in enumerate(fracture): #receiver dde
            r_dde['dsl'] = DD[k][0]
            r_dde['dsh'] = DD[k+len(fracture)][0]
            r_dde['dnn'] = DD[k+2*len(fracture)][0]

def cal_stress_disp(fractures:list,respara:dict,fibre):
    '''
    calculate synthetic stress and displacements in DAS fibre induced by the displacement discontinuity
    fracture: fracture objects that contains displacement discountinuities
    respara: reservoir parameters
    fibre: DAS fibre objects that contains global coordinates for each DAS channel
    '''
    youngs = 2.0*respara['ShearModulus']*(1.0+respara['PoissonRatio'])
    for chn in fibre: #receiver dde
        # initialize stresses and displacements for the channel
        sxx = 0
        syy = 0
        szz = 0
        sxy = 0
        sxz = 0
        syz = 0
        exx = 0
        eyy = 0
        ezz = 0
        uxx = 0
        uyy = 0
        uzz = 0
        for fracture in fractures:
            for dde in fracture:
                cos_beta = np.cos(np.deg2rad(dde['strike']))
                cos_2beta = np.cos(np.deg2rad(2*dde['strike']))
                sin_beta = np.sin(np.deg2rad(dde['strike']))
                sin_2beta = np.sin(np.deg2rad(2*dde['strike']))
                #print(cos_beta, cos_2beta, sin_beta, sin_2beta)
                x1 = (chn['X']-dde['X'])*cos_beta+(chn['Y']-dde['Y'])*sin_beta
                x2 = chn['Z']-dde['Z']
                x3 = -(chn['X']-dde['X'])*sin_beta+(chn['Y']-dde['Y'])*cos_beta
                a = dde['dl']/2.0
                b = dde['dh']/2.0
                #calculate coefficient
                Cr = respara['ShearModulus']/(4.0*np.pi*(1.0-respara['PoissonRatio']))
                r1 = np.sqrt((x1-a)**2+(x2-b)**2+x3**2)
                r2 = np.sqrt((x1-a)**2+(x2+b)**2+x3**2)
                r3 = np.sqrt((x1+a)**2+(x2-b)**2+x3**2)
                r4 = np.sqrt((x1+a)**2+(x2+b)**2+x3**2)
                #I,1
                J1 = np.log(r1+x2-b)-np.log(r2+x2+b)-np.log(r3+x2-b)+np.log(r4+x2+b)
                #I,2
                J2 = np.log(r1+x1-a)-np.log(r2+x1-a)-np.log(r3+x1+a)+np.log(r4+x1+a)
                #I,3
                J3 = -np.arctan((x1-a)*(x2-b)/(x3*r1))+np.arctan((x1-a)*(x2+b)/(x3*r2))+\
                     np.arctan((x1+a)*(x2-b)/(x3*r3))-np.arctan((x1+a)*(x2+b)/(x3*r4))
                #I,11
                J4 = (x1-a)/(r1*(r1+x2-b))-(x1-a)/(r2*(r2+x2+b))-(x1+a)/(r3*(r3+x2-b))+(x1+a)/(r4*(r4+x2+b))
                #I,22
                J5 = (x2-b)/(r1*(r1+x1-a))-(x2+b)/(r2*(r2+x1-a))-(x2-b)/(r3*(r3+x1+a))+(x2+b)/(r4*(r4+x1+a))
                #I,33
                J6 = (x1-a)*(x2-b)*(x3**2+r1**2)/(r1*(x3**2+(x1-a)**2)*(x3**2+(x2-b)**2))-\
                     (x1-a)*(x2+b)*(x3**2+r2**2)/(r2*(x3**2+(x1-a)**2)*(x3**2+(x2+b)**2))-\
                     (x1+a)*(x2-b)*(x3**2+r3**2)/(r3*(x3**2+(x1+a)**2)*(x3**2+(x2-b)**2))+\
                     (x1+a)*(x2+b)*(x3**2+r4**2)/(r4*(x3**2+(x1+a)**2)*(x3**2+(x2+b)**2))
                #I,12
                J7 = 1/r1-1/r2-1/r3+1/r4
                #I,13
                J8 = x3/(r1*(r1+x2-b))-x3/(r2*(r2+x2+b))-x3/(r3*(r3+x2-b))+x3/(r4*(r4+x2+b))
                #I,23
                J9 = x3/(r1*(r1+x1-a))-x3/(r2*(r2+x1-a))-x3/(r3*(r3+x1+a))+x3/(r4*(r4+x1+a))
                #I,32
                J9_1 = x3/(r1*(r1+x1-a))-x3/(r2*(r2+x1-a))-x3/(r3*(r3+x1+a))+x3/(r4*(r4+x1+a))
                #I,111
                J10 = (-(r1+x2-b)*((x1-a)**2-r1**2)-(x1-a)**2*r1)/(r1**3*(r1+x2-b)**2)-\
                      (-(r2+x2+b)*((x1-a)**2-r2**2)-(x1-a)**2*r2)/(r2**3*(r2+x2+b)**2)-\
                      (-(r3+x2-b)*((x1+a)**2-r3**2)-(x1+a)**2*r3)/(r3**3*(r3+x2-b)**2)+\
                      (-(r4+x2+b)*((x1+a)**2-r4**2)-(x1+a)**2*r4)/(r4**3*(r1+x2+b)**2)
                #I,211
                J11 = (-(x1-a)/r1**3)-(-(x1-a)/r2**3)-(-(x1+a)/r3**3)+(-(x1+a)/r4**3)
                #I,311
                J12 = (-(x1-a)*x3*(2*r1+x2-b))/(r1**3*(r1+x2-b)**2)-\
                      (-(x1-a)*x3*(2*r2+x2+b))/(r2**3*(r2+x2+b)**2)-\
                      (-(x1+a)*x3*(2*r3+x2-b))/(r3**3*(r3+x2-b)**2)+\
                      (-(x1+a)*x3*(2*r4+x2+b))/(r4**3*(r4+x2+b)**2)
                #I,311
                J13 = (-(x2-b))/r1**3-(-(x2+b))/r2**3-(-(x2-b))/r3**3+(-(x2+b))/r4**3
                #I,122
                J14 = (-(r1+x1-a)*((x2-b)**2-r1**2)-(x2-b)**2*r1)/(r1**3*(r1+x1-a)**2)-\
                      (-(r2+x1-a)*((x2+b)**2-r2**2)-(x2+b)**2*r2)/(r2**3*(r2+x1-a)**2)-\
                      (-(r3+x1+a)*((x2-b)**2-r3**2)-(x2-b)**2*r3)/(r3**3*(r3+x1+a)**2)+\
                      (-(r4+x1+a)*((x2+b)**2-r4**2)-(x2+b)**2*r4)/(r4**3*(r4+x1+a)**2)
                #I,222
                J15 = (-(x2-b)*x3*(2*r1+x1-a))/(r1**3*(r1+x1-a)**2)-\
                      (-(x2+b)*x3*(2*r2+x1-a))/(r2**3*(r2+x1-a)**2)-\
                      (-(x2-b)*x3*(2*r3+x1+a))/(r3**3*(r3+x1+a)**2)+\
                      (-(x2+b)*x3*(2*r4+x1+a))/(r4**3*(r4+x1+a)**2)
                #I,322
                J16 = (-(r1+x2-b)*(x3**2-r1**2)-x3**2*r1)/(r1**3*(r1+x2-b)**2)-\
                      (-(r2+x2+b)*(x3**2-r2**2)-x3**2*r2)/(r2**3*(r2+x2+b)**2)-\
                      (-(r3+x2-b)*(x3**2-r3**2)-x3**2*r3)/(r3**3*(r3+x2-b)**2)+\
                      (-(r4+x2+b)*(x3**2-r4**2)-x3**2*r4)/(r4**3*(r4+x2+b)**2)
                #I,233
                J17 = (-(r1+x1-a)*(x3**2-r1**2)-x3**2*r1)/(r1**3*(r1+x1-a)**2)-\
                      (-(r2+x1-a)*(x3**2-r2**2)-x3**2*r2)/(r2**3*(r2+x1-a)**2)-\
                      (-(r3+x1+a)*(x3**2-r3**2)-x3**2*r3)/(r3**3*(r3+x1+a)**2)+\
                      (-(r4+x1+a)*(x3**2-r4**2)-x3**2*r4)/(r4**3*(r4+x1+a)**2)
                #I,333
                J18 = (-x3*(x1-a)*(x2-b))*((x3**2+(x1-a)**2)**2*(x3**2+(x2-b)**2+2*r1**2)+\
                      (x3**2+(x2-b)**2)**2*(x3**2+(x1-a)**2+2*r1**2))/(r1**3*(x3**2+(x1-a)**2)**2*(x3**2+(x2-b)**2)**2)-\
                      (-x3*(x1-a)*(x2+b))*((x3**2+(x1-a)**2)**2*(x3**2+(x2+b)**2+2*r2**2)+\
                      (x3**2+(x2+b)**2)**2*(x3**2+(x1-a)**2+2*r2**2))/(r2**3*(x3**2+(x1-a)**2)**2*(x3**2+(x2+b)**2)**2)-\
                      (-x3*(x1+a)*(x2-b))*((x3**2+(x1+a)**2)**2*(x3**2+(x2-b)**2+2*r3**2)+\
                      (x3**2+(x2-b)**2)**2*(x3**2+(x1+a)**2+2*r3**2))/(r3**3*(x3**2+(x1+a)**2)**2*(x3**2+(x2-b)**2)**2)+\
                      (-x3*(x1+a)*(x2+b))*((x3**2+(x1+a)**2)**2*(x3**2+(x2+b)**2+2*r4**2)+\
                      (x3**2+(x2+b)**2)**2*(x3**2+(x1+a)**2+2*r4**2))/(r4**3*(x3**2+(x1+a)**2)**2*(x3**2+(x2+b)**2)**2)
                #I,123
                J19 = (-x3/r1**3)-(-x3/r2**3)-(-x3/r3**3)+(-x3/r4**3)

                #calculate stress and displacement in local coordinates
                SS11 = Cr*dde['dsl']*(2*J8-x3*J10)+Cr*dde['dsh']*(2*respara['PoissonRatio']*J9-x3*J11)+\
                       Cr*dde['dnn']*(J6+(1-2*respara['PoissonRatio'])*J5-x3*J12)
                SS22 = Cr*dde['dsl']*(2*respara['PoissonRatio']*J8-x3*J13)+Cr*dde['dsh']*(2*J9-x3*J14)+\
                       Cr*dde['dnn']*(J6+(1-2*respara['PoissonRatio'])*J4-x3*J15)
                SS33 = Cr*dde['dsl']*(-x3*J16)+Cr*dde['dsh']*(-x3*J17)+Cr*dde['dnn']*(J6-x3*J18)
                SS12 = Cr*dde['dsl']*((1-respara['PoissonRatio'])*J9-x3*J11)+\
                       Cr*dde['dsh']*((1-respara['PoissonRatio'])*J8-x3*J13)+\
                       Cr*dde['dnn']*(-(1-2*respara['PoissonRatio'])*J7-x3*J19)
                SS13 = Cr*dde['dsl']*(J6+respara['PoissonRatio']*J5-x3*J12)+\
                       Cr*dde['dsh']*(-respara['PoissonRatio']*J7-x3*J19)+Cr*dde['dnn']*(-x3*J16)
                SS23 = Cr*dde['dsl']*(-respara['PoissonRatio']*J7-x3*J19)+\
                       Cr*dde['dsh']*(J6+respara['PoissonRatio']*J4-x3*J15)+Cr*dde['dnn']*(-x3*J17)
                
                U1 = (2*(1-respara['PoissonRatio'])*dde['dsl']*J3-\
                     (1-2*respara['PoissonRatio'])*dde['dnn']*J1-\
                     x3*(dde['dsl']*J4+dde['dsh']*J7+dde['dnn']*J8))/(8*np.pi*(1-respara['PoissonRatio']))
                U2 = (2*(1-respara['PoissonRatio'])*dde['dsh']*J3-\
                     (1-2*respara['PoissonRatio'])*dde['dnn']*J2-\
                     x3*(dde['dsl']*J7+dde['dsh']*J5+dde['dnn']*J9))/(8*np.pi*(1-respara['PoissonRatio']))
                U3 = (2*(1-respara['PoissonRatio'])*dde['dnn']*J3+\
                     (1-2*respara['PoissonRatio'])*(dde['dsl']*J1+\
                     dde['dsh']*J2)-x3*(dde['dsl']*J8+dde['dsh']*J9+dde['dnn']*J6))/(8*np.pi*(1-respara['PoissonRatio']))
                #accumulate stress and displacement in the global coordinates
                sxx += cos_beta**2*SS11-sin_2beta*SS13+sin_beta**2*SS33
                syy += sin_beta**2*SS11+sin_2beta*SS13+cos_beta**2*SS33
                szz += SS22
                sxy += sin_beta*cos_beta*SS11+cos_2beta*SS13-sin_beta*cos_beta*SS33
                sxz += cos_beta*SS12-sin_beta*SS23
                syz += sin_beta*SS12+cos_beta*SS23
                
                uxx += cos_beta*U1-sin_beta*U3
                uyy += sin_beta*U1+cos_beta*U3
                uzz += U2
        # append calculated stresses and displacements to fibre
        exx = 1/youngs*(sxx-respara['PoissonRatio']*(syy+szz))
        eyy = 1/youngs*(syy-respara['PoissonRatio']*(sxx+szz))
        ezz = 1/youngs*(szz-respara['PoissonRatio']*(syy+sxx))
        chn['SXX'].append(sxx)
        chn['SYY'].append(syy)
        chn['SZZ'].append(szz)
        chn['SXY'].append(sxy)
        chn['SXZ'].append(sxz)
        chn['SYZ'].append(syz)
        chn['EXX'].append(exx)
        chn['EYY'].append(eyy)
        chn['EZZ'].append(ezz)
        chn['UXX'].append(uxx)
        chn['UYY'].append(uyy)
        chn['UZZ'].append(uzz)

        
def fracture_plot_location(fracture):
    x = []
    y = []
    z = []
    for dde in fracture:
        x.append(dde['X'])
        y.append(dde['Y'])
        z.append(dde['Z'])
#     x = np.asarray(x)
#     y = np.asarray(y)
#     z = np.asarray(z)
    fig = plt.Figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.scatter(x,y)
    plt.show()
        

def fracture_plot_disp(fracture,tot_l,tot_h,c_x,c_y,c_z,opt):
    '''
    Plot a color image of the displacement of each dde in a fracture, i.e., the aperture map of the fracture
    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # initialize img
    dl = fracture[0]['dl']
    dh = fracture[0]['dh']
    l_n = round(tot_l/fracture[0]['dl'])
    h_n = round(tot_h/fracture[0]['dh'])
    img_dsl = np.zeros((l_n,h_n))
    img_dsh = np.zeros((l_n,h_n))
    img_dnn = np.zeros((l_n,h_n))
    ###
    #transform fracture plane back to zero strike
    sk = -fracture[0]['strike']
    dp = -fracture[0]['dip']
    yw = -fracture[0]['yaw']
    
    cos_SK = np.cos(np.deg2rad(sk))
    sin_SK = np.sin(np.deg2rad(sk))
    
    cos_DP = np.cos(np.deg2rad(dp))
    sin_DP = np.sin(np.deg2rad(dp))
    
    cos_YW = np.cos(np.deg2rad(yw))
    sin_YW = np.sin(np.deg2rad(yw))
    
    SK = np.array([[cos_SK,sin_SK,0.0],[-sin_SK,cos_SK,0.0],[0.0,0.0,1.0]])
    DP = np.array([[1.0,0.0,0.0],[0.0,cos_DP,sin_DP],[0.0,-sin_DP,cos_DP]])
    YW = np.array([[cos_YW,sin_YW,0.0],[-sin_YW,cos_YW,0.0],[0.0,0.0,1.0]])
    
    AA = np.matmul(np.matmul(SK,DP),YW)
    ###
    for dde in fracture:
        if dde['strike'] != 0.0 or dde['dip'] != 0.0 or dde['yaw'] != 0.0:
            gl = np.array([[dde['X']-c_x],[dde['Y']-c_y],[dde['Z']-c_z]])
            # coordinates trasform
            lo = np.matmul(AA,gl)
            x = lo[0][0]+c_x
            y = lo[1][0]+c_y
            z = lo[2][0]+c_z
            i = round((2*(x-c_x)+tot_l)/(2*dl)+0.5)-1
            j = round((2*(z-c_z)+tot_h)/(2*dh)+0.5)-1
            img_dsl[i,j] = dde['dsl']*1e3
            img_dsh[i,j] = dde['dsh']*1e3
            img_dnn[i,j] = dde['dnn']*1e3
            #print('{},{},x:{:0.2f},y:{:0.2f},z:{:0.2f}'.format(i,j,dde['X'],dde['Y'],dde['Z']))
            #print('{},{},x:{:0.2f},y:{:0.2f},z:{:0.2f}'.format(i,j,x,y,z))
        else:
            i = round((2*(dde['X']-c_x)+tot_l)/(2*dl)+0.5)-1
            j = round((2*(dde['Z']-c_z)+tot_h)/(2*dh)+0.5)-1
            img_dsl[i,j] = dde['dsl']*1e3
            img_dsh[i,j] = dde['dsh']*1e3
            img_dnn[i,j] = dde['dnn']*1e3
    fig = plt.figure(figsize=(4*tot_l/tot_h,4))
    ax = fig.add_subplot(111)
    l = np.linspace(-tot_l/2,tot_l/2,l_n)
    h = np.linspace(-tot_h/2,tot_h/2,h_n)
    L, H = np.meshgrid(l,h)
    if opt == 0:
        vmax = np.max(np.abs(img_dsl))
        levels = MaxNLocator(nbins=200).tick_values(0,vmax)
        img = ax.contourf(L,H,np.abs(img_dsl.T),cmap='viridis',levels=levels,extend='both')
    elif opt == 1:
        vmax = np.max(np.abs(img_dsh))
        levels = MaxNLocator(nbins=200).tick_values(0,vmax)
        img = ax.contourf(L,H,np.abs(img_dsh.T),cmap='viridis',levels=levels,extend='both')
    else:
        vmax = np.max(np.abs(img_dnn))
        levels = MaxNLocator(nbins=200).tick_values(0,vmax)
        img = ax.contourf(L,H,np.abs(img_dnn.T),cmap='viridis',levels=levels,extend='both')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.2, pad=0.05)
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.set_ylabel('Displacement(mm)', labelpad=15)
    ax.set_xlabel('Length(m)')
    ax.set_ylabel('Height(m)')
    plt.show()
    
def fracture_plot_disp_ellipse(fracture,tot_l,tot_h,c_x,c_y,c_z,opt:int=2):
    '''
    Plot a color image of the displacement of each dde in a fracture, i.e., the aperture map of the fracture
    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # initialize img
    dl = fracture[0]['dl']
    dh = fracture[0]['dh']
    l_n = round(tot_l/fracture[0]['dl'])
    h_n = round(tot_h/fracture[0]['dh'])
    for dde in fracture:
        i = round((2*(dde['X']-c_x)+tot_l)/(2*dde['dl'])+0.5)-1
        j = round((2*(dde['Z']-c_z)+tot_h)/(2*dde['dh'])+0.5)-1
        tester = (dde['X']-c_x)**2/(tot_l/2)**2+(dde['Z']-c_z)**2/(tot_h/2)**2
        if tester <= 1:
            img_dsl[i,j] = dde['dsl']*1e3
            img_dsh[i,j] = dde['dsh']*1e3
            img_dnn[i,j] = dde['dnn']*1e3
        else:
            img_dsl[i,j] = 0.0
            img_dsh[i,j] = 0.0
            img_dnn[i,j] = 0.0
    fig = plt.figure(figsize=(4*tot_l/tot_h,4))
    ax = fig.add_subplot(111)
    l = np.linspace(-tot_l/2,tot_l/2,int(tot_l/fracture[0]['dl'])) # Time
    h = np.linspace(-tot_h/2,tot_h/2,int(tot_h/fracture[0]['dh'])) # channel number
    L, H = np.meshgrid(l,h)
    if opt == 0:
        vmax = np.max(np.abs(img_dsl))
        levels = MaxNLocator(nbins=200).tick_values(0,vmax)
        img = ax.contourf(L,H,np.abs(img_dsl.T),cmap='viridis',levels=levels,extend='both')
    elif opt == 1:
        vmax = np.max(np.abs(img_dsh))
        levels = MaxNLocator(nbins=200).tick_values(0,vmax)
        img = ax.contourf(L,H,np.abs(img_dsh.T),cmap='viridis',levels=levels,extend='both')
    else:
        vmax = np.max(np.abs(img_dnn))
        levels = MaxNLocator(nbins=200).tick_values(0,vmax)
        img = ax.contourf(L,H,np.abs(img_dnn.T),cmap='viridis',levels=levels,extend='both')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.2, pad=0.05)
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.set_ylabel('Displacement(mm)', labelpad=15)
    ax.set_xlabel('Length(m)')
    ax.set_ylabel('Height(m)')
    plt.show()
    
def fracture_plot_stress_dd_time(fractures,snn,ssl,c_x,c_y,c_z):
    '''
    Plot a time series of the discontinuity components of the center elements of the fault
    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import FormatStrFormatter
    # initialize img
    dsl = np.zeros(len(fractures))
    dsh = np.zeros(len(fractures))
    dnn = np.zeros(len(fractures))
    t = np.linspace(1,90,90)
    for i, fracture in enumerate(fractures):
        for dde in fracture[0]:
            if dde['X']-c_x<1e-3 and dde['Y']-c_y<1e-3 and dde['Z']<1e-3:
                dsl[i] = dde['dsl']*1e3
                dsh[i] = dde['dsh']*1e3
                dnn[i] = dde['dnn']*1e3
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4,4))
    fig.subplots_adjust(hspace=0.1)
    
    ax[0].plot(t,snn/1e6,color='r',label='Net Pressure', linewidth=2.0)
    ax[0].plot(t,-ssl/1e6,color='b',label='Shear Stress', linewidth=2.0)
    ax[1].plot(t, dsl, color='r', label='Dsl', linewidth=2.0)
    ax[1].plot(t, dsh, color='b', label='Dsh', linewidth=2.0)
    ax[1].plot(t, dnn, color='green', label='Dnn', linewidth=2.0)
    
    ax[0].set_ylabel('Stress(MPa)',fontsize=12)
    ax[0].set_xlim(0,90)
    ax[0].set_ylim(-3,3)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[0].tick_params('both',labelsize=12)
    ax[0].legend(ncol=2,loc='lower center',fontsize=10,frameon=False)
    
    ax[1].set_ylim(-0.2,0.6)
    ax[1].set_xlabel('Time(min)',fontsize=12)
    ax[1].set_ylabel('Disp(mm)',fontsize=12)
    ax[1].tick_params('both',labelsize=12)
    ax[1].legend(ncol=3,loc='lower center',fontsize=10,frameon=False)
    plt.show()
    fig.savefig('stress_dd_plot.png', dpi=150, transparent=True, bbox_inches = 'tight')

    
def fibre_plot(fibre,opt,gl,scale,figsize):
    '''
    Plot a color image of the quantity stored in the fibre channels
    fibre: fibre that stores quantities
    opt: options
    '''
    #calculate distance between fiber channels
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig = plt.figure(figsize=(figsize[0],figsize[1]))
    ax = fig.add_subplot(111)
    # initialize img
    if opt == 'SXX':
        _img = np.zeros((len(fibre[0]['SXX']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['SXX']
        t = np.linspace(0,len(chn['SXX']),len(chn['SXX'])) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],len(fibre)) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/1e6/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,-_img.T/1e6,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Stress(MPa)',labelpad=15,rotation=270)
    elif opt == 'SYY':
        _img = np.zeros((len(fibre[0]['SYY']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['SYY']
        t = np.linspace(0,len(chn['SYY']),len(chn['SYY'])) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],len(fibre)) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/1e6/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,-_img.T/1e6,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Stress(MPa)',labelpad=15,rotation=270)
    elif opt == 'SZZ':
        _img = np.zeros((len(fibre[0]['SZZ']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['SZZ']
        t = np.linspace(0,len(chn['SZZ']),len(chn['SZZ'])) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],len(fibre)) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/1e6/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,-_img.T/1e6,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Stress(MPa)',labelpad=15,rotation=270)
    elif opt == 'SXY':
        _img = np.zeros((len(fibre[0]['SXY']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['SXY']
        t = np.linspace(0,len(chn['SXY']),len(chn['SXY'])) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],len(fibre)) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/1e6/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,_img.T/1e6,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Stress(MPa)',labelpad=15,rotation=270)
    elif opt == 'SXZ':
        _img = np.zeros((len(fibre[0]['SXZ']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['SXZ']
        t = np.linspace(0,len(chn['SXZ']),len(chn['SXZ'])) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],len(fibre)) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/1e6/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,_img.T/1e6,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Stress(MPa)',labelpad=15,rotation=270)
    elif opt == 'SYZ':
        _img = np.zeros((len(fibre[0]['SYZ']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['SYZ']
        t = np.linspace(0,len(chn['SYZ']),len(chn['SYZ'])) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],len(fibre)) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/1e6/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,_img.T/1e6,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Stress(MPa)',labelpad=15,rotation=270)
    elif opt == 'UXX':
        _img = np.zeros((len(fibre[0]['UXX']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['UXX']
        t = np.linspace(0,len(chn['UXX']),len(chn['UXX'])) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],len(fibre)) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))*1e3/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,_img.T*1e3,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Displacement(mm)',labelpad=15,rotation=270)
    elif opt == 'UYY':
        _img = np.zeros((len(fibre[0]['UYY']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['UYY']
#         if smooth == True:
#             _img = np.delete(_img,slice(round(_img.shape[1]/2-sn),round(_img.shape[1]/2+sn)),1)
        t = np.linspace(0,len(chn['UYY']),np.shape(_img)[0]) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],np.shape(_img)[1]) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))*1e3/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,_img.T*1e3,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Displacement(mm)',labelpad=15,rotation=270)
    elif opt == 'UZZ':
        _img = np.zeros((len(fibre[0]['UZZ']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['UZZ']
        t = np.linspace(0,len(chn['UZZ']),len(chn['UZZ'])) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],len(fibre)) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))*1e3/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,_img.T*1e3,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Displacement(mm)',labelpad=15,rotation=270)
    elif opt == 'EXX_U':
        _img = np.zeros((len(fibre[0]['UXX']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['UXX']
        for i in range(_img.shape[0]):
            for j in range(_img.shape[1]-gl):
                _img[i][j] = (_img[i][j+gl] - _img[i][j])/gl*1e6
        t = np.linspace(0,len(chn['UXX']),np.shape(_img)[0]) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],np.shape(_img)[1]) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,-_img.T,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$',labelpad=15)
    elif opt == 'EYY_U':
        _img = np.zeros((len(fibre[0]['UYY']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['UYY']
        for i in range(_img.shape[0]):
            for j in range(_img.shape[1]-gl):
                _img[i][j] = (_img[i][j+gl] - _img[i][j])/gl*1e6
        t = np.linspace(0,len(chn['UYY']),np.shape(_img)[0]) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],np.shape(_img)[1]) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,_img.T,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$',labelpad=15)
    elif opt == 'EZZ_U':
        _img = np.zeros((len(fibre[0]['UZZ']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['UZZ']
        for i in range(_img.shape[0]):
            for j in range(_img.shape[1]-gl):
                _img[i][j] = (_img[i][j+gl] - _img[i][j])/gl*1e6
        t = np.linspace(0,len(chn['UYY']),np.shape(_img)[0]) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],np.shape(_img)[1]) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,-_img.T,cmap='bwr',levels=levels,extend='both') #sign convention change
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$',labelpad=15)
    elif opt == 'EXX':
        _img = np.zeros((len(fibre[0]['EXX']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['EXX']
        _img = _img*1e6
        t = np.linspace(0,len(chn['EXX']),len(chn['EXX'])) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],len(fibre)) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,-_img.T,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$',labelpad=15)
    elif opt == 'EYY':
        _img = np.zeros((len(fibre[0]['EYY']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['EYY']
        _img = _img*1e6
        t = np.linspace(0,len(chn['EYY']),len(chn['EYY'])) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],len(fibre)) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,-_img.T,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$',labelpad=15)
    elif opt == 'EZZ':
        _img = np.zeros((len(fibre[0]['EZZ']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['EZZ']
        _img = _img*1e6
        t = np.linspace(0,len(chn['EZZ']),len(chn['EZZ'])) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],len(fibre)) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,-_img.T,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$',labelpad=15)
    elif opt == 'EXX_Rate':
        _img = np.zeros((len(fibre[0]['UXX']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['EXX']
        _img = np.diff(_img,axis=0)*1e6
        t = np.linspace(0,len(chn['EXX']),len(chn['EXX'])-1) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],len(fibre)) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,-_img.T,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$/min',labelpad=15)
    elif opt == 'EYY_Rate':
        _img = np.zeros((len(fibre[0]['EYY']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['EYY']
        _img = np.diff(_img,axis=0)*1e6
        t = np.linspace(0,len(chn['EYY']),len(chn['EYY'])-1) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],len(fibre)) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,-_img.T,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$/min',labelpad=15)
    elif opt == 'EZZ_Rate':
        _img = np.zeros((len(fibre[0]['EZZ']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['EZZ']
        _img = np.diff(_img,axis=0)*1e6
        t = np.linspace(0,len(chn['EZZ']),len(chn['EZZ'])-1) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],len(fibre)) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,-_img.T,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$/min',labelpad=15)
    elif opt == 'EXX_U_Rate':
        _img = np.zeros((len(fibre[0]['UXX']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['UXX']
        for i in range(_img.shape[0]):
            for j in range(_img.shape[1]-gl):
                _img[i][j] = (_img[i][j+gl] - _img[i][j])/gl*1e6
        _img = np.diff(_img,axis=0)
        t = np.linspace(0,len(chn['UXX']),np.shape(_img)[0]) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],np.shape(_img)[1]) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,_img.T,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$/min',labelpad=15)
    elif opt == 'EYY_U_Rate':
        _img = np.zeros((len(fibre[0]['UYY']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['UYY']
        for i in range(_img.shape[0]):
            for j in range(_img.shape[1]-gl):
                _img[i][j] = (_img[i][j+gl] - _img[i][j])/gl*1e6
        _img = np.diff(_img,axis=0)
        t = np.linspace(0,len(chn['UYY']),np.shape(_img)[0]) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],np.shape(_img)[1]) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,_img.T,cmap='bwr',levels=levels,extend='both')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$/min',labelpad=15)
    elif opt == 'EZZ_U_Rate':
        _img = np.zeros((len(fibre[0]['UZZ']),len(fibre)))
        for i, chn in enumerate(fibre):
            _img[:,i] = chn['UZZ']
        for i in range(_img.shape[0]):
            for j in range(_img.shape[1]-gl):
                _img[i][j] = (_img[i][j+gl] - _img[i][j])/gl*1e6
        _img = np.diff(_img,axis=0)
        t = np.linspace(0,len(chn['UZZ']),np.shape(_img)[0]) # Time
        chn = np.linspace(0,len(fibre)*chn['dl'],np.shape(_img)[1]) # channel number
        T, CHN = np.meshgrid(t,chn)
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        
        img = ax.contourf(T,CHN,-_img.T,cmap='bwr',levels=levels,extend='both') #sign convention change
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$/min',labelpad=15)
    ax.invert_yaxis()
    ax.set_xlabel('Time(min)')
    ax.set_ylabel('Fibre Length(m)')
    plt.tight_layout()
    fig.savefig(opt,dpi=300,transparent=True)
    plt.show
    
    
def plane_plot(plane,opt,scale):
    '''
    Plot a color image of the quantity stored in the fibre channels
    fibre: fibre that stores quantities
    opt: options
    '''
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    example_node = plane[0]
    size = example_node['size']
    node_size = example_node['node_size']
    _img = np.zeros((int(size[0]/node_size[0]),int(size[1]/node_size[1]))) # initialize _img
    l = np.linspace(-size[0]/2,size[0]/2,int(size[0]/node_size[0]))
    w = np.linspace(-size[0]/2,size[1]/2,int(size[1]/node_size[1]))
    L,W = np.meshgrid(l,w)
    if opt == 'SXX':
        for node in plane:
            _img[node['row'],node['col']] = node['SXX'][0]
        vmax = np.max(np.abs(_img))/1e6/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        img = ax.contourf(L,W,_img.T/1e6,cmap='bwr',levels=levels,extend='both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Stress(MPa)',labelpad=15,rotation=270)
    elif opt == 'SYY':
        for node in plane:
            _img[node['row'],node['col']] = node['SYY'][0]
        vmax = np.max(np.abs(_img))/1e6/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        img = ax.contourf(L,W,_img.T/1e6,cmap='bwr',levels=levels,extend='both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Stress(MPa)',labelpad=15,rotation=270)
    elif opt == 'SZZ':
        for node in plane:
            _img[node['row'],node['col']] = node['SZZ'][0]
        vmax = np.max(np.abs(_img))/1e6/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        img = ax.contourf(L,W,_img.T/1e6,cmap='bwr',levels=levels,extend='both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Stress(MPa)',labelpad=15,rotation=270)
    elif opt == 'SXY':
        for node in plane:
            _img[node['row'],node['col']] = node['SXY'][0]
        vmax = np.max(np.abs(_img))/1e6/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        img = ax.contourf(L,W,_img.T/1e6,cmap='bwr',levels=levels,extend='both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Stress(MPa)',labelpad=15,rotation=270)
    elif opt == 'SXZ':
        for node in plane:
            _img[node['row'],node['col']] = node['SXZ'][0]
        vmax = np.max(np.abs(_img))/1e6/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        img = ax.contourf(L,W,_img.T/1e6,cmap='bwr',levels=levels,extend='both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Stress(MPa)',labelpad=15,rotation=270)
    elif opt == 'SYZ':
        for node in plane:
            _img[node['row'],node['col']] = node['SYZ'][0]
        vmax = np.max(np.abs(_img))/1e6/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        img = ax.contourf(L,W,_img.T/1e6,cmap='bwr',levels=levels,extend='both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Stress(MPa)',labelpad=15,rotation=270)
    elif opt == 'UXX':
        for node in plane:
            _img[node['row'],node['col']] = node['UXX'][0]
        vmax = np.max(np.abs(_img))*1e3/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        img = ax.contourf(L,W,_img.T*1e3,cmap='bwr',levels=levels,extend='both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Displacement(mm)',labelpad=15,rotation=270)
    elif opt == 'UYY':
        for node in plane:
            _img[node['row'],node['col']] = node['UYY'][0]
        vmax = np.max(np.abs(_img))*1e3/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        img = ax.contourf(L,W,_img.T*1e3,cmap='bwr',levels=levels,extend='both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Displacement(mm)',labelpad=15,rotation=270)
    elif opt == 'UZZ':
        for node in plane:
            _img[node['row'],node['col']] = node['UZZ'][0]
        vmax = np.max(np.abs(_img))*1e3/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        img = ax.contourf(L,W,_img.T*1e3,cmap='bwr',levels=levels,extend='both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('Displacement(mm)',labelpad=15,rotation=270)
    elif opt == 'EXX_U':
        for node in plane:
            _img[node['row'],node['col']] = node['UXX'][0]
        _img = np.diff(_img.T,axis=1)/node_size[0]*1e6
        l1 = np.linspace(-size[0]/2,size[0]/2,int(size[0]/node_size[0])-1)
        w1 = np.linspace(-size[1]/2,size[1]/2,int(size[1]/node_size[1]))
        L1,W1 = np.meshgrid(l1,w1)
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        img = ax.contourf(L1,W1,_img,cmap='bwr',levels=levels,extend='both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$',labelpad=15)
    elif opt == 'EYY_U':
        for node in plane:
            _img[node['row'],node['col']] = node['UYY'][0]
        _img = np.diff(_img.T,axis=0)/node_size[0]*1e6
        l1 = np.linspace(-size[0]/2,size[0]/2,int(size[0]/node_size[0]))
        w1 = np.linspace(-size[1]/2,size[1]/2,int(size[1]/node_size[1])-1)
        L1,W1 = np.meshgrid(l1,w1)
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        img = ax.contourf(L1,W1,_img,cmap='bwr',levels=levels,extend='both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$',labelpad=15)
    elif opt == 'EZZ_U':
        for node in plane:
            _img[node['row'],node['col']] = node['UZZ'][0]
        _img = np.diff(_img.T,axis=0)/node_size[0]*1e6
        l1 = np.linspace(-size[0]/2,size[0]/2,int(size[0]/node_size[0]))
        w1 = np.linspace(-size[1]/2,size[1]/2,int(size[1]/node_size[1])-1)
        L1,W1 = np.meshgrid(l1,w1)
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        img = ax.contourf(L1,W1,_img,cmap='bwr',levels=levels,extend='both') #sign convention change
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$',labelpad=15)
    elif opt == 'EXX':
        for node in plane:
            _img[node['row'],node['col']] = node['EXX'][0]
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        img = ax.contourf(L,W,_img.T,cmap='bwr',levels=levels,extend='both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$',labelpad=15)
    elif opt == 'EYY':
        for node in plane:
            _img[node['row'],node['col']] = node['EYY'][0]
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        img = ax.contourf(L,W,_img.T,cmap='bwr',levels=levels,extend='both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$',labelpad=15)
    elif opt == 'EZZ':
        for node in plane:
            _img[node['row'],node['col']] = node['EZZ'][0]
        vmax = np.max(np.abs(_img))/scale
        levels = MaxNLocator(nbins=200).tick_values(-vmax,vmax)
        img = ax.contourf(L,W,_img.T,cmap='bwr',levels=levels,extend='both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel('$\mu\epsilon$',labelpad=15)
    ax.invert_yaxis()
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Width (m)')
    plt.show
    
def fracture_plot_location(fracture,number,save):
    x = []
    y = []
    z = []
    for dde in fracture:
        x.append(dde['X'])
        y.append(dde['Y'])
        z.append(dde['Z'])
#     x = np.asarray(x)
#     y = np.asarray(y)
#     z = np.asarray(z)
    fig = plt.Figure(figsize=(8,8))
    plt.xlim(-200,200)
    plt.ylim(-200,200)
    plt.gca().invert_yaxis()
    plt.scatter(x,y,c='turquoise')
    plt.show
    if save == True:
        plt.savefig(str(number)+'.png',transparent=True,dpi=150)

    