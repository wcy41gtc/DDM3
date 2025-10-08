from fibre import *
from fracture import *

def cal_dd(fractures:list):
    '''
    calculate the displacement discontinuity based on stress conditions
    '''
    if len(fractures) == 0:  # if elements list is empty
        sys.exit(
            'Error! No fracture input was found.')
    for fracture in fractures:
        coef_slsl = np.zeros((len(fracture.elements),len(fracture.elements)))
        coef_slsh = np.zeros((len(fracture.elements),len(fracture.elements)))
        coef_slnn = np.zeros((len(fracture.elements),len(fracture.elements)))
        coef_shsl = np.zeros((len(fracture.elements),len(fracture.elements)))
        coef_shsh = np.zeros((len(fracture.elements),len(fracture.elements)))
        coef_shnn = np.zeros((len(fracture.elements),len(fracture.elements)))
        coef_nnsl = np.zeros((len(fracture.elements),len(fracture.elements)))
        coef_nnsh = np.zeros((len(fracture.elements),len(fracture.elements)))
        coef_nnnn = np.zeros((len(fracture.elements),len(fracture.elements)))
        Ssl = np.zeros((len(fracture.elements),1))
        Ssh = np.zeros((len(fracture.elements),1))
        Snn = np.zeros((len(fracture.elements),1))
        for i, r_dde in enumerate(fracture.elements):
            Ssl[i] = r_dde.Ssl
            Ssh[i] = r_dde.Ssh
            Snn[i] = r_dde.Snn
            for j, dde in enumerate(fracture.elements):
                gamma = r_dde.strike - dde.strike   # difference in strike for local coordinates
                cos_gamma = np.cos(np.deg2rad(gamma))
                cos_2gamma = np.cos(np.deg2rad(2 * gamma))
                sin_gamma = np.sin(np.deg2rad(gamma))
                sin_2gamma = np.sin(np.deg2rad(2 * gamma))
                cos_beta = np.cos(np.deg2rad(r_dde.strike))
                cos_2beta = np.cos(np.deg2rad(2 * r_dde.strike))
                sin_beta = np.sin(np.deg2rad(r_dde.strike))
                sin_2beta = np.sin(np.deg2rad(2 * r_dde.strike))
                x1 = (r_dde.x - dde.x) * cos_beta + (r_dde.y - dde.y) * sin_beta
                x2 = r_dde.z - dde.z
                x3 = -(r_dde.x - dde.x) * sin_beta + (r_dde.y - dde.y) * cos_beta
                a = dde.dl / 2.0
                b = dde.dh / 2.0
                #calculate coefficient
                Cr = fracture.ShearModulus/(4.0*np.pi*(1.0-fracture.PoissonRatio))
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
                                     cos_2gamma*(J6+fracture.PoissonRatio*J5-x3*J12)+sin_gamma*cos_gamma*(-x3*J16))
                coef_slsh[i,j] = Cr*(-sin_gamma*cos_gamma*(2*fracture.PoissonRatio*J9-x3*J11)+\
                                     cos_2gamma*(-fracture.PoissonRatio*J7-x3*J19)+sin_gamma*cos_gamma*(-x3*J17))
                coef_slnn[i,j] = Cr*(-sin_gamma*cos_gamma*(J6+(1-2*fracture.PoissonRatio)*J5-x3*J12)+\
                                     cos_2gamma*(-x3*J16)+sin_gamma*cos_gamma*(J6-x3*J18))
                #shsl, shsh, shnn
                coef_shsl[i,j] = Cr*(-sin_gamma*((1-fracture.PoissonRatio)*J9-x3*J11)+\
                                     cos_gamma*(-fracture.PoissonRatio*J7-x3*J19))
                coef_shsh[i,j] = Cr*(-sin_gamma*((1-fracture.PoissonRatio)*J8-x3*J13)+\
                                     cos_gamma*(J6+fracture.PoissonRatio*J4-x3*J15))
                coef_shnn[i,j] = Cr*(-sin_gamma*(-(1-2*fracture.PoissonRatio)*J7-x3*J19)+\
                                     cos_gamma*(-x3*J17))
                #nnsl, nnsh, nnnn
                coef_nnsl[i,j] = Cr*(sin_gamma**2*(2*J8-x3*J10)-\
                                     2*sin_gamma*cos_gamma*(J6+fracture.PoissonRatio*J5-x3*J12)+cos_gamma**2*(-x3*J16))
                coef_nnsh[i,j] = Cr*(sin_gamma**2*(2*fracture.PoissonRatio*J9-x3*J11)-\
                                     2*sin_gamma*cos_gamma*(-fracture.PoissonRatio*J7-x3*J19)+cos_gamma**2*(-x3*J17))
                coef_nnnn[i,j] = Cr*(sin_gamma**2*(J6+(1-2*fracture.PoissonRatio)*J5-x3*J12)-\
                                     2*sin_gamma*cos_gamma*(-x3*J16)+cos_gamma**2*(J6-x3*J18))
        S = np.vstack((Ssl,Ssh,Snn))    # assemble coef matrix and displacement column vector
        coef = np.vstack((np.hstack((coef_slsl, coef_slsh, coef_slnn)),
                          np.hstack((coef_shsl, coef_shsh, coef_shnn)),
                          np.hstack((coef_nnsl, coef_nnsh, coef_nnnn))))
        coef[np.isnan(coef)] = 0.0
        DD_solution = np.linalg.solve(coef, S)    # solve for DD
        for k, r_dde in enumerate(fracture.elements):    # assign DD back to dde in fracture
            r_dde.dsl = DD_solution[k][0]
            r_dde.dsh = DD_solution[k+len(fracture)][0]
            r_dde.dnn = DD_solution[k+2*len(fracture)][0]


def cal_fibre(fractures:list,fibres:list):
    '''
    calculate synthetic stress and displacements in DAS fibre induced by the displacement discontinuity
    fracture: fracture objects that contains displacement discountinuities
    respara: reservoir parameters
    fibre: DAS fibre objects that contains global coordinates for each DAS channel
    '''
    for fibre in fibres:
        for chn in fibre: #receiver dde
            # initialize stresses and displacements for the channel
            sxx = 0
            syy = 0
            szz = 0
            sxy = 0
            sxz = 0
            syz = 0
            uxx = 0
            uyy = 0
            uzz = 0
            for fracture in fractures:
                for dde in fracture.elements:
                    cos_beta = np.cos(np.deg2rad(dde.strike))
                    cos_2beta = np.cos(np.deg2rad(2*dde.strike))
                    sin_beta = np.sin(np.deg2rad(dde.strike))
                    sin_2beta = np.sin(np.deg2rad(2*dde.strike))
                    x1 = (chn.x-dde.x)*cos_beta+(chn.y-dde.y)*sin_beta
                    x2 = chn.z-dde.z
                    x3 = -(chn.x-dde.x)*sin_beta+(chn.y-dde.y)*cos_beta
                    a = dde.dl/2.0
                    b = dde.dh/2.0
                    #calculate coefficient
                    Cr = fracture.ShearModulus/(4.0*np.pi*(1.0-fracture.PoissonRatio))
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
                    SS11 = Cr*dde.dsl*(2*J8-x3*J10)+Cr*dde.dsh*(2*fracture.PoissonRatio*J9-x3*J11)+\
                           Cr*dde.dnn*(J6+(1-2*fracture.PoissonRatio)*J5-x3*J12)
                    SS22 = Cr*dde.dsl*(2*fracture.PoissonRatio*J8-x3*J13)+Cr*dde.dsh*(2*J9-x3*J14)+\
                           Cr*dde.dnn*(J6+(1-2*fracture.PoissonRatio)*J4-x3*J15)
                    SS33 = Cr*dde.dsl*(-x3*J16)+Cr*dde.dsh*(-x3*J17)+Cr*dde.dnn*(J6-x3*J18)
                    SS12 = Cr*dde.dsl*((1-fracture.PoissonRatio)*J9-x3*J11)+\
                           Cr*dde.dsh*((1-fracture.PoissonRatio)*J8-x3*J13)+\
                           Cr*dde.dnn*(-(1-2*fracture.PoissonRatio)*J7-x3*J19)
                    SS13 = Cr*dde.dsl*(J6+fracture.PoissonRatio*J5-x3*J12)+\
                           Cr*dde.dsh*(-fracture.PoissonRatio*J7-x3*J19)+Cr*dde.dnn*(-x3*J16)
                    SS23 = Cr*dde.dsl*(-fracture.PoissonRatio*J7-x3*J19)+\
                           Cr*dde.dsh*(J6+fracture.PoissonRatio*J4-x3*J15)+Cr*dde.dnn*(-x3*J17)
                    if abs(SS11) > 1e10:
                        SS11=0
                    if abs(SS22) > 1e10:
                        SS22=0
                    if abs(SS33) > 1e10:
                        SS33=0
                    if abs(SS12) > 1e10:
                        SS12=0
                    if abs(SS13) > 1e10:
                        SS13=0
                    if abs(SS23) > 1e10:
                        SS23=0
                    U1 = (2*(1-fracture.PoissonRatio)*dde.dsl*J3-\
                         (1-2*fracture.PoissonRatio)*dde.dnn*J1-\
                         x3*(dde.dsl*J4+dde.dsh*J7+dde.dnn*J8))/(8*np.pi*(1-fracture.PoissonRatio))
                    U2 = (2*(1-fracture.PoissonRatio)*dde.dsh*J3-\
                         (1-2*fracture.PoissonRatio)*dde.dnn*J2-\
                         x3*(dde.dsl*J7+dde.dsh*J5+dde.dnn*J9))/(8*np.pi*(1-fracture.PoissonRatio))
                    U3 = (2*(1-fracture.PoissonRatio)*dde.dnn*J3+\
                         (1-2*fracture.PoissonRatio)*(dde.dsl*J1+\
                         dde.dsh*J2)-x3*(dde.dsl*J8+dde.dsh*J9+dde.dnn*J6))/(8*np.pi*(1-fracture.PoissonRatio))

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
            chn.SXX.append(sxx)
            chn.SYY.append(syy)
            chn.SZZ.append(szz)
            chn.SXY.append(sxy)
            chn.SXZ.append(sxz)
            chn.SYZ.append(syz)
            chn.UXX.append(uxx)
            chn.UYY.append(uyy)
            chn.UZZ.append(uzz)
