import numpy as np
import scipy.special as ss

class MembraneHydro:
    
    def __init__ (self, _lattice_parameter,
                  _lattice_n_x, _lattice_n_y,
                  _temperature, _viscosity, _alpha_over_a,
                  _quadrature_area="circle", _n_quadrature_points=4):
        
        self.d = _lattice_parameter

        self.area_per_particle = 0.5 * np.sqrt(3.0) * (self.d * self.d)
        
        self.n_x = _lattice_n_x
        self.n_y = _lattice_n_y

        self.alpha = _alpha_over_a * self.d
        self.alpha2 = self.alpha * self.alpha
        self.alpha3 = self.alpha2 * self.alpha

        self.eta = _viscosity
        self.T = _temperature

        self.kT = 1.38064852e-2 * self.T

        self.BuildPlane ()

        self.quad_point_x = None
        self.quad_point_y = None
        self.quad_weight = None

        if _quadrature_area == "circle":
            
            self.r_c = np.sqrt (self.area_per_particle / np.pi)
            # self.r_c = 0.5 * self.d

            print ("Cut-off radius = ", self.r_c)

            self.BuildLocalIntegrationGrid_Circle (_n_quadrature_points)

        elif _quadrature_area == "hexagon":
            
            self.r_c = 0.5 * self.d
            self.BuildLocalIntegrationGrid_Hexagon (_n_quadrature_points)

        else:

            raise ValueError("The integration area '" + _quadrature_area +"' is undefined!")

        self.zeta = None
        self.D = None
        self.A = None
        
    def BuildPlane (self):
        
        self.n_particles = self.n_x * self.n_y

        self.pos = np.zeros ([self.n_particles, 2])
        
        _l = self.d * 0.5 * np.sqrt(3.0)
        
        for j in range(self.n_y):
            for i in range (self.n_x):
                
                ind = i + j * self.n_x
                
                fi = float(i)
                fj = float(j)
                fj_mod_2 = float(j % 2)
                
                self.pos[ind, 0] = fi * self.d + 0.5 * fj_mod_2 * self.d
                self.pos[ind, 1] = fj * _l
        
        self.L_x = float (self.n_x) * self.d
        self.L_y = float (self.n_y) * _l

        self.hL_x = 0.5 * self.L_x
        self.hL_y = 0.5 * self.L_y

    def BuildLocalIntegrationGrid_Hexagon (self, n_div):

        s3_3 = np.sqrt(3.0) / 3.0

        e10 = np.array ([self.r_c, - self.r_c * s3_3]) / float (n_div)
        e20 = np.array ([self.r_c, self.r_c * s3_3]) / float (n_div)

        _quad_point_x = []
        _quad_point_y = []

        dA = s3_3 * (self.r_c / float (n_div)) ** 2

        for sector in range (6):

            th = float (sector) * np.pi / 3.0

            ct = np.cos (th)
            st = np.sin (th)

            e1 = np.array ([e10[0] * ct - e10[1] * st, e10[0] * st + e10[1] * ct])
            e2 = np.array ([e20[0] * ct - e20[1] * st, e20[0] * st + e20[1] * ct])

            for i in range (n_div):
                for j in range (n_div - i):

                    r = (1.0 / 3.0 + i) * e1 + (1.0 / 3.0 + j) * e2

                    _quad_point_x.append(r[0])
                    _quad_point_y.append(r[1])

                for j in range(n_div - i - 1):

                    r = (2.0 / 3.0 + i) * e1 + (2.0 / 3.0 + j) * e2

                    _quad_point_x.append(r[0])
                    _quad_point_y.append(r[1])


        self.quad_point_x = np.array(_quad_point_x)
        self.quad_point_y = np.array(_quad_point_y)

        self.quad_weight = np.ones_like(self.quad_point_x) * dA

    def BuildLocalIntegrationGrid_Circle (self, n_quad_points):
        
        #self.lr = np.linspace (0.0, self.r_c, self.n_div)
        #self.dlr = self.lr[1] - self.lr[0]

        #self.lth = np.linspace (0.0, 2.0 * np.pi, 2 * self.n_div)
        #self.dlth = self.lth[1] - self.lth[0]

        ##Local integration grid
        #self.x_grid = np.tensordot (self.lr, np.cos (self.lth), axes = 0)
        #self.y_grid = np.tensordot (self.lr, np.sin (self.lth), axes = 0)
        
        #Guass-Legendre quadrature on the unit disk (by KyoungJoong Kim and ManSuk Song)
        
        if n_quad_points == 1:

            w_1 = 3.141592653589793
            x_1 = 0.0
            
            self.quad_point_x = np.array ([x_1]) * self.r_c

            self.quad_point_y = np.array ([x_1]) * self.r_c
                                            
            self.quad_weight = np.array ([w_1]) * self.r_c * self.r_c

        elif n_quad_points == 4:

            w_1 = 0.785398163397448
            x_1 = 0.5
            
            self.quad_point_x = np.array ([x_1, -x_1, -x_1, x_1]) * self.r_c

            self.quad_point_y = np.array ([x_1, x_1, -x_1, -x_1]) * self.r_c
                                            
            self.quad_weight = np.array ([w_1, w_1, w_1, w_1]) * self.r_c * self.r_c

        elif n_quad_points == 8:

            w_1 = 0.732786462492640
            w_2 = 0.052611700904808
            x_1 = 0.650115167343736
            x_2 = 0.888073833977115
            
            self.quad_point_x = np.array ([x_1, 0.0, -x_1, 0.0, x_2, -x_2, -x_2, x_2]) * self.r_c

            self.quad_point_y = np.array ([0.0, x_1, 0.0, -x_1, x_2, x_2, -x_2, -x_2]) * self.r_c
                                            
            self.quad_weight = np.array ([w_1, w_1, w_1, w_1, w_2, w_2, w_2, w_2]) * self.r_c * self.r_c
            
        elif n_quad_points == 12:

            w_1 = 0.232710566932577
            w_2 = 0.387077796006226
            w_3 = 0.165609800458645
            
            x_1 = 0.866025403784439
            x_2 = 0.322914992067400
            x_3 = 0.644171310389465
            
            self.quad_point_x = np.array ([x_1, 0.0, -x_1, 0.0, x_2, -x_2, -x_2, x_2, x_3, -x_3, -x_3, x_3]) * self.r_c

            self.quad_point_y = np.array ([0.0, x_1, 0.0, -x_1, x_2, x_2, -x_2, -x_2, x_3, x_3, -x_3, -x_3]) * self.r_c
                                            
            self.quad_weight = np.array ([w_1, w_1, w_1, w_1, w_2, w_2, w_2, w_2, w_3, w_3, w_3, w_3]) * self.r_c * self.r_c

        elif n_quad_points == 20:

            w_1 = 0.071488826617391
            w_2 = 0.327176874928167
            w_3 = 0.005591341512851
            w_4 = 0.190570560169519

            x_1 = 0.952458896434417
            x_2 = 0.415187657878755
            x_3 = 0.834794942216211
            x_4 = 0.740334457173511
            y_4 = 0.379016937530835

            self.quad_point_x = np.array(
                [x_1, 0.0, -x_1, 0.0, x_2, 0.0, -x_2, 0.0, x_3, -x_3, -x_3, x_3, x_4, -x_4, -x_4, x_4, y_4, y_4, -y_4,
                 -y_4]) * self.r_c

            self.quad_point_y = np.array(
                [0.0, x_1, 0.0, -x_1, 0.0, x_2, 0.0, -x_2, x_3, x_3, -x_3, -x_3, y_4, y_4, -y_4, -y_4, x_4, -x_4, -x_4,
                 x_4]) * self.r_c

            self.quad_weight = np.array(
                [w_1, w_1, w_1, w_1, w_2, w_2, w_2, w_2, w_3, w_3, w_3, w_3, w_4, w_4, w_4, w_4, w_4, w_4, w_4,
                 w_4]) * self.r_c * self.r_c

        elif n_quad_points == 44:

            x_1 = 0.252863797091293
            x_2 = 0.989746802511614
            x_3 = 0.577728928444958
            x_4 = 0.873836956645035
            x_5 = 0.689299380791136
            x_6 = 0.597614304667208
            x_7 = 0.375416824626170
            x_8 = 0.883097111318591
            y_8 = 0.365790800400663
            x_9 = 0.707438744960070
            y_9 = 0.293030722710664

            w_1 = 0.125290208564338
            w_2 = 0.016712625496982
            w_3 = 0.109500391126365
            w_4 = 0.066237455796397
            w_5 = 0.026102860184358
            w_6 = 0.066000934661100
            w_7 = 0.127428372681720
            w_8 = 0.042523065826681
            w_9 = 0.081539591616413

            self.quad_point_x = np.array(
                [x_1, 0.0, -x_1, 0.0, x_2, 0.0, -x_2, 0.0, x_3, 0.0, -x_3, 0.0, x_4, 0.0, -x_4, 0.0,
                 x_5, -x_5, -x_5, x_5, x_6, -x_6, -x_6, x_6, x_7, -x_7, -x_7, x_7,
                 x_8, -x_8, -x_8, x_8, y_8, y_8, -y_8, -y_8,
                 x_9, -x_9, -x_9, x_9, y_9, y_9, -y_9, -y_9]) * self.r_c

            self.quad_point_y = np.array(
                [0.0, x_1, 0.0, -x_1, 0.0, x_2, 0.0, -x_2, 0.0, x_3, 0.0, -x_3, 0.0, x_4, 0.0, -x_4,
                 x_5, x_5, -x_5, -x_5, x_6, x_6, -x_6, -x_6, x_7, x_7, -x_7, -x_7,
                 y_8, y_8, -y_8, -y_8, x_8, -x_8, -x_8, x_8,
                 y_9, y_9, -y_9, -y_9, x_9, -x_9, -x_9, x_9]) * self.r_c

            self.quad_weight = np.array(
                [w_1, w_1, w_1, w_1, w_2, w_2, w_2, w_2, w_3, w_3, w_3, w_3, w_4, w_4, w_4, w_4,
                 w_5, w_5, w_5, w_5, w_6, w_6, w_6, w_6, w_7, w_7, w_7, w_7,
                 w_8, w_8, w_8, w_8, w_8, w_8, w_8, w_8,
                 w_9, w_9, w_9, w_9, w_9, w_9, w_9, w_9]) * self.r_c * self.r_c

        elif n_quad_points == 72:

            w_1 = 0.082558858859169
            x_1 = 0.204668989256100

            w_2 = 0.009721593541193
            x_2 = 0.992309839464756

            w_3 = 0.061920685878045
            x_3 = 0.740931035494388
            
            w_4 = 0.079123279187043
            x_4 = 0.477987648986077

            w_5 = 0.087526733002317
            x_5 = 0.306138805262459

            w_6 = 0.057076811471306
            x_6 = 0.524780156099700

            w_7 = 0.020981864256888
            x_7 = 0.921806074110042
            y_7 = 0.310920075968188

            w_8 = 0.015226392255721
            x_8 = 0.790235832571934
            y_8 = 0.579897645710646

            w_9 = 0.033136884897617
            x_9 = 0.725790566968788
            y_9 = 0.525045580895713

            w_10 = 0.044853730819348
            x_10 = 0.788230650371813
            y_10 = 0.290244481132460

            w_11 = 0.065321481701811
            x_11 = 0.584894890453686
            y_11 = 0.264317463415838

            w_12 = 0.024214746797802
            x_12 = 0.909637445684200
            y_12 = 0.09257113237088

            self.quad_point_x = np.array(
                [x_1, 0.0, -x_1, 0.0, x_2, 0.0, -x_2, 0.0, x_3, 0.0, -x_3, 0.0, x_4, 0.0, -x_4, 0.0,
                 x_5, -x_5, -x_5, x_5, x_6, -x_6, -x_6, x_6,
                 x_7, -x_7, -x_7, x_7, y_7, y_7, -y_7, -y_7,
                 x_8, -x_8, -x_8, x_8, y_8, y_8, -y_8, -y_8,
                 x_9, -x_9, -x_9, x_9, y_9, y_9, -y_9, -y_9,
                 x_10, -x_10, -x_10, x_10, y_10, y_10, -y_10, -y_10,
                 x_11, -x_11, -x_11, x_11, y_11, y_11, -y_11, -y_11,
                 x_12, -x_12, -x_12, x_12, y_12, y_12, -y_12, -y_12]) * self.r_c

            self.quad_point_y = np.array(
                [0.0, x_1, 0.0, -x_1, 0.0, x_2, 0.0, -x_2, 0.0, x_3, 0.0, -x_3, 0.0, x_4, 0.0, -x_4,
                 x_5, x_5, -x_5, -x_5, x_6, x_6, -x_6, -x_6,
                 y_7, y_7, -y_7, -y_7, x_7, -x_7, -x_7, x_7,
                 y_8, y_8, -y_8, -y_8, x_8, -x_8, -x_8, x_8,
                 y_9, y_9, -y_9, -y_9, x_9, -x_9, -x_9, x_9,
                 y_10, y_10, -y_10, -y_10, x_10, -x_10, -x_10, x_10,
                 y_11, y_11, -y_11, -y_11, x_11, -x_11, -x_11, x_11,
                 y_12, y_12, -y_12, -y_12, x_12, -x_12, -x_12, x_12]) * self.r_c

            self.quad_weight = np.array(
                [w_1, w_1, w_1, w_1, w_2, w_2, w_2, w_2, w_3, w_3, w_3, w_3, w_4, w_4, w_4, w_4,
                 w_5, w_5, w_5, w_5, w_6, w_6, w_6, w_6,
                 w_7, w_7, w_7, w_7, w_7, w_7, w_7, w_7,
                 w_8, w_8, w_8, w_8, w_8, w_8, w_8, w_8,
                 w_9, w_9, w_9, w_9, w_9, w_9, w_9, w_9,
                 w_10, w_10, w_10, w_10, w_10, w_10, w_10, w_10,
                 w_11, w_11, w_11, w_11, w_11, w_11, w_11, w_11,
                 w_12, w_12, w_12, w_12, w_12, w_12, w_12, w_12]) * self.r_c * self.r_c

        else:

            raise ValueError ("No set of points/weights for the choice of " + str(n_quad_points) + " quadrature point!")

    def Wrap (self, dx, dy):

        wdx = dx - np.floor(dx / self.L_x + 0.5) * self.L_x
        wdy = dy - np.floor(dy / self.L_y + 0.5) * self.L_y

        return wdx, wdy

    def CalcFrictionTensor (self):
        ## Friction coefficient tensor
        
        # this corresponds to a normalized Gaussian used for point velocities
        # it is necessary in the sense that the "momentum" of the distributed velocity be equal
        # to the actual momentum of the particle

        factor = self.eta * self.area_per_particle / (4.0 * np.sqrt (np.pi) * self.alpha3)

        self.zeta = np.zeros ([self.n_particles, self.n_particles])

        for i in range (self.n_particles):
            
            # self.zeta[i, i] = factor * 8.0 * (np.pi ** 1.5) * self.eta * self.alpha *\
            #                  (np.exp (-self.xi_c) * self.xi_c * ( ss.iv (0, self.xi_c) - ss.iv (1, self.xi_c) ) )
            
            for j in range(i, self.n_particles):
                
                integrand = 0.0
                
                for k in range (self.quad_weight.shape[0]):

                    dx = self.pos[j, 0] + self.quad_point_x[k] - self.pos[i, 0]
                    dy = self.pos[j, 1] + self.quad_point_y[k] - self.pos[i, 1]
                    
                    dx, dy = self.Wrap(dx, dy)

                    r2_ij = dx * dx + dy * dy
                    
                    xi = 0.125 * (r2_ij / self.alpha2)

                    if xi < 700.0:

                        f1 = np.exp(-xi)
                        f2 = f1 * ss.iv(0, xi)
                        f3 = f1 * ss.iv(1, xi)

                    else:
                        #Asymptotic expansion as x -> inf
                        f1 = np.sqrt (1.0 / xi)
                        f2 = f1 / np.sqrt (2.0 * np.pi) + f1 ** 3 / (8.0 * np.sqrt (2.0 * np.pi))
                        f3 = f1 / np.sqrt (2.0 * np.pi) - 3.0 * f1 ** 3 / (8.0 * np.sqrt (2.0 * np.pi))

                    integrand += ((1.0 - 2.0 * xi) * f2 + 2.0 * xi * f3) * self.quad_weight[k]

                self.zeta[i, j] = factor * integrand
                
                self.zeta[j, i] = self.zeta[i, j]

    def CalcDiffusionTensorDirect(self, cut_off=np.inf):
        ## Diffusion coefficient tensor

        # this corresponds to a normalized Gaussian used for point velocities
        # it is necessary in the sense that the "momentum" of the distributed velocity be equal
        # to the actual momentum of the particle

        # The surface integral of velocity distribution is W, which is A_p * v_p

        factor = self.kT / (8.0 * np.sqrt (np.pi) * self.alpha * self.eta * self.area_per_particle)

        co2 = cut_off * cut_off

        self.D = np.zeros([self.n_particles, self.n_particles])

        for i in range(self.n_particles):

            for j in range(i, self.n_particles):

                integrand = 0.0

                for k in range (self.quad_weight.shape[0]):

                    dx = self.pos[j, 0] + self.quad_point_x[k] - self.pos[i, 0]
                    dy = self.pos[j, 1] + self.quad_point_y[k] - self.pos[i, 1]

                    dx, dy = self.Wrap(dx, dy)

                    r2_ij = dx * dx + dy * dy

                    if r2_ij < co2:

                        xi = 0.125 * (r2_ij / self.alpha2)

                        if xi < 700.0:

                            f1 = np.exp(-xi)
                            f2 = f1 * ss.iv(0, xi)

                        else:
                            #Asymptotic expansion as x -> inf
                            f1 = np.sqrt (1.0 / xi)
                            f2 = f1 / np.sqrt (2.0 * np.pi) + f1 ** 3 / (8.0 * np.sqrt (2.0 * np.pi))

                        integrand += f2 * self.quad_weight[k]


                self.D[i, j] = factor * integrand

                self.D[j, i] = self.D[i, j]

    def CalcDiffusionTensor (self):
        self.CalcFrictionTensor ()

        self.D = self.kT * np.linalg.inv (self.zeta)

    ## Calculates the derivate of the diffusion tensor with respect to the position of the particle with index "ind".
    def CalcDiffusionTensorDerivative (self, ind, cut_off=np.inf):

        eps = 1.0e-6

        self.CalcDiffusionTensorDirect(cut_off)
        D0 = np.copy (self.D)
        pos0 = np.copy (self.pos[ind, :])

        self.pos[ind, 0] += eps * self.d

        self.CalcDiffusionTensorDirect(cut_off)

        Dx1 = np.copy (self.D)

        self.pos[ind, :] = pos0[:]

        self.pos[ind, 1] += eps * self.d

        self.CalcDiffusionTensorDirect(cut_off)

        Dy1 = np.copy (self.D)

        dD_dx = (Dx1 - D0) / eps
        dD_dy = (Dy1 - D0) / eps

        self.pos[ind, :] = pos0[:]

        self.D = D0

        return dD_dx, dD_dy

    def CalcRndDecomposedTensor(self, cut_off=np.inf):

        if self.D.any () is None:
            self.CalcDiffusionTensorDirect(cut_off)

        self.A = np.linalg.cholesky(self.D)

    def GetHistograms (self, M, _n_bins):
        ## Histogram of a matrix M based on interparticle distances
        
        r = np.linspace (0.0, np.sqrt (self.L_x * self.L_x + self.L_y * self.L_y), _n_bins)

        dr = r[1] - r[0]

        n_hist = np.zeros (_n_bins)

        M_hist = np.zeros (_n_bins)
        M_min_hist = np.ones (_n_bins) * 1.0e10
        M_max_hist = np.ones (_n_bins) * -1.0e10

        for i in range (self.n_particles):
            
            for j in range (i, self.n_particles):
                
                dx = self.pos[j, 0] - self.pos[i, 0]
                dy = self.pos[j, 1] - self.pos[i, 1]

                dx, dy = self.Wrap (dx, dy)

                r_ij = np.sqrt (dx * dx + dy * dy)

                ind = np.round (r_ij / dr).astype('int')
                
                M_hist[ind] += M[i, j]

                if (M[i, j] < M_min_hist[ind]):
                    M_min_hist[ind] = M[i, j]

                if (M[i, j] > M_max_hist[ind]):
                    M_max_hist[ind] = M[i, j]

                n_hist[ind] += 1.0

        nz_ind = np.nonzero (n_hist)
        
        M_mean_hist = M_hist[nz_ind] / n_hist[nz_ind]

        return r[nz_ind], M_mean_hist, M_min_hist[nz_ind], M_max_hist[nz_ind]

    def GetDiffusionHistogramDirect (self, use_compensation=False, n_compensation=1):

        ## Histogram of diffusion coefficient based on interparticle distances

        _n_bins = 2000

        r = np.linspace(0.0, np.sqrt(self.L_x * self.L_x + self.L_y * self.L_y), _n_bins)

        dr = r[1] - r[0]

        n_hist = np.zeros(_n_bins)

        for i in range(self.n_particles):

            for j in range(i, self.n_particles):

                dx = self.pos[j, 0] - self.pos[i, 0]
                dy = self.pos[j, 1] - self.pos[i, 1]

                r_ij = np.sqrt(dx * dx + dy * dy)

                ind = np.round(r_ij / dr).astype('int')

                n_hist[ind] += 1.0

        nz_ind = np.nonzero(n_hist)

        r_list = r[nz_ind]

        r_list = np.linspace(0.0, np.sqrt(self.L_x * self.L_x + self.L_y * self.L_y), 10000)

        D_list = []

        factor = self.kT / (8.0 * np.sqrt (np.pi) * self.alpha * self.eta * self.area_per_particle)

        if (use_compensation):

            dth = 2.0 * np.pi / float(n_compensation)

            th = np.arange(n_compensation) * dth

            R_ring = 3.0 * self.alpha
            # R_ring = self.d
            # R_ring = self.r_c

            alpha2_comp = (self.alpha * 1.0) ** 2

            pos_comp = np.array([R_ring * np.cos(th), R_ring * np.sin(th)])

            rr = np.linspace(0.0, np.sqrt(alpha2_comp) * 1000.0, 10000)
            sig1 = np.exp(- 0.125 * (rr ** 2 / self.alpha2))

            sig2 = np.zeros_like(rr)

            for n in range(n_compensation):

                sig2 += np.exp (- 0.125 * (((rr - pos_comp[0, n]) ** 2 + pos_comp[1, n] ** 2) / alpha2_comp)) / float (n_compensation)

            I1 = np.trapz (sig1 * 2.0 * np.pi * rr, rr)
            I2 = np.trapz (sig2 * 2.0 * np.pi * rr, rr)

            ratio = I1 / I2

            sig = sig1 - ratio * sig2

        for _r in r_list:

            I_center = 0.0

            for k in range(self.quad_weight.shape[0]):

                dx = _r + self.quad_point_x[k]
                dy = self.quad_point_y[k]

                r2 = dx * dx + dy * dy

                xi = 0.125 * (r2 / self.alpha2)

                if xi < 700.0:

                    f1 = np.exp(-xi)
                    f2 = f1 * ss.iv(0, xi)

                else:
                    # Asymptotic expansion as x -> inf
                    f1 = np.sqrt(1.0 / xi)
                    f2 = f1 / np.sqrt(2.0 * np.pi) + f1 ** 3 / (8.0 * np.sqrt(2.0 * np.pi))

                I_center += f2 * self.quad_weight[k]


            if use_compensation:

                I_ring = 0.0

                for n in range (n_compensation):

                    for k in range(self.quad_weight.shape[0]):

                        dx = _r + self.quad_point_x[k] - pos_comp[0, n]
                        dy = self.quad_point_y[k] - pos_comp[1, n]

                        r2 = dx * dx + dy * dy

                        xi = 0.125 * (r2 / alpha2_comp)

                        if xi < 700.0:

                            f1 = np.exp(-xi)
                            f2 = f1 * ss.iv(0, xi)

                        else:
                            # Asymptotic expansion as x -> inf
                            f1 = np.sqrt(1.0 / xi)
                            f2 = f1 / np.sqrt(2.0 * np.pi) + f1 ** 3 / (8.0 * np.sqrt(2.0 * np.pi))

                        I_ring += f2 * self.quad_weight[k]

                D_list.append (factor * (I_center - ratio * I_ring / float (n_compensation)))

            else:

                D_list.append (factor * I_center)

        if use_compensation:

            return r_list, np.array (D_list), rr, sig

        else:

            return r_list, np.array (D_list)


    def FindNeighbors (self, ind0):

        nei_ind = []

        for i in range(self.n_particles):

            dx = self.pos[i, 0] - self.pos[ind0, 0]
            dy = self.pos[i, 1] - self.pos[ind0, 1]

            dx, dy = self.Wrap(dx, dy)

            r_ij = np.sqrt(dx * dx + dy * dy)

            if np.abs (r_ij - self.d) < 1.0e-6 * self.d:

                nei_ind.append (i)

        if (len (nei_ind) == 0):
            raise ValueError ("No neighbors found!")

        th = []

        v = np.zeros (2)

        for n_ind in nei_ind:

            v[0] = self.pos[n_ind, 0] - self.pos[ind0, 0]
            v[1] = self.pos[n_ind, 1] - self.pos[ind0, 1]

            v[0], v[1] = self.Wrap(v[0], v[1])

            th.append (np.arctan2 (v[1], v[0]))

        sorted_n_ind = np.argsort (np.array (th))

        return np.array (nei_ind)[sorted_n_ind]

