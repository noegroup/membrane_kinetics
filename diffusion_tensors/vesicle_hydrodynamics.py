import numpy as np
import scipy.special as ss


class VesicleHydro:

    def __init__(self, _lattice_parameter, _level_of_tess,
                 _temperature, _viscosity, _alpha_over_a,
                 _quadrature_area="circle", _n_quadrature_points=4):
        """
        :type _quadrature_area: str
        :type _n_quadrature_points: int
        """

        self.d = _lattice_parameter
        self.n_angle_div = 100000

        self.level_of_tess = _level_of_tess

        self.alpha = _alpha_over_a * self.d
        
        self.alpha2 = self.alpha * self.alpha
        self.alpha3 = self.alpha2 * self.alpha

        self.eta = _viscosity
        self.T = _temperature

        self.kT = 1.38064852e-2 * self.T

        self.R = 0
        self.n_particles = 0
        self.pos = None
        self.normal = None

        self.build_sphere()

        self.R2 = self.R * self.R
        self.R3 = self.R2 * self.R

        self.area_per_particle = 4.0 * np.pi * self.R ** 2 / float(self.n_particles)

        self.r_c = np.sqrt(self.area_per_particle / np.pi)

        print("Number of particles = ", self.n_particles)
        print("Vesicle radius = ", self.R)
        print("Area per particle = ", self.area_per_particle)
        print("Effective radius of each particle = ", self.r_c)

        self.quad_point_x = None
        self.quad_point_y = None
        self.quad_weight = None

        if _quadrature_area == "circle":

            self.build_local_integration_grid_circle(_n_quadrature_points)

        elif _quadrature_area == "hexagon":

            self.build_local_integration_grid_hexagon(_n_quadrature_points)

        else:

            raise ValueError("The integration area '" + _quadrature_area + "' is undefined!")

        self.v_r = None
        self.sig_r = None

        self.calc_surface_stress()

        self.zeta = None
        self.D = None
        self.A = None

    def build_sphere(self):

        def push_point(v):

            v /= np.sqrt(np.sum(v * v))

        def tess_tri(p_old, _i, _j, _k, level):

            p = np.zeros([3, 3])

            p[0, :] = 0.5 * (p_old[_i, :] + p_old[_j, :])
            p[1, :] = 0.5 * (p_old[_j, :] + p_old[_k, :])
            p[2, :] = 0.5 * (p_old[_k, :] + p_old[_i, :])

            push_point(p[0, :])
            push_point(p[1, :])
            push_point(p[2, :])

            n_points = p_old.shape[0]

            p_old = np.append(p_old, p, axis=0)

            m = n_points
            n = n_points + 1
            q = n_points + 2

            if level > 1:
                p_old = tess_tri(p_old, _i, m, q, level - 1)
                p_old = tess_tri(p_old, m, _j, n, level - 1)
                p_old = tess_tri(p_old, q, n, _k, level - 1)
                p_old = tess_tri(p_old, n, m, q, level - 1)

            return p_old

        self.pos = np.zeros([12, 3])

        a = np.arctan(0.5)

        b = 2.0 * np.pi / 5.0
        c = np.pi / 5.0

        th = np.array([0.5 * np.pi, a, a, a, a, a, -a, -a, -a, -a, -a, -0.5 * np.pi])

        phi = np.array(
            [0.0, 0.0, b, 2.0 * b, 3.0 * b, 4.0 * b, c, c + b, c + 2.0 * b, c + 3.0 * b, c + 4.0 * b, 0.0])

        self.pos[:, 0] = np.cos(th[:]) * np.cos(phi[:])
        self.pos[:, 1] = np.cos(th[:]) * np.sin(phi[:])
        self.pos[:, 2] = np.sin(th[:])

        if self.level_of_tess > 1:
            self.pos = tess_tri(self.pos, 0, 1, 2, self.level_of_tess - 1)

            self.pos = tess_tri(self.pos, 0, 2, 3, self.level_of_tess - 1)
            self.pos = tess_tri(self.pos, 0, 3, 4, self.level_of_tess - 1)
            self.pos = tess_tri(self.pos, 0, 4, 5, self.level_of_tess - 1)
            self.pos = tess_tri(self.pos, 0, 5, 1, self.level_of_tess - 1)

            self.pos = tess_tri(self.pos, 1, 6, 2, self.level_of_tess - 1)
            self.pos = tess_tri(self.pos, 2, 6, 7, self.level_of_tess - 1)

            self.pos = tess_tri(self.pos, 2, 7, 3, self.level_of_tess - 1)
            self.pos = tess_tri(self.pos, 3, 7, 8, self.level_of_tess - 1)

            self.pos = tess_tri(self.pos, 3, 8, 4, self.level_of_tess - 1)
            self.pos = tess_tri(self.pos, 4, 8, 9, self.level_of_tess - 1)

            self.pos = tess_tri(self.pos, 4, 9, 5, self.level_of_tess - 1)
            self.pos = tess_tri(self.pos, 5, 9, 10, self.level_of_tess - 1)

            self.pos = tess_tri(self.pos, 5, 10, 1, self.level_of_tess - 1)
            self.pos = tess_tri(self.pos, 1, 10, 6, self.level_of_tess - 1)

            self.pos = tess_tri(self.pos, 11, 6, 7, self.level_of_tess - 1)
            self.pos = tess_tri(self.pos, 11, 7, 8, self.level_of_tess - 1)
            self.pos = tess_tri(self.pos, 11, 8, 9, self.level_of_tess - 1)
            self.pos = tess_tri(self.pos, 11, 9, 10, self.level_of_tess - 1)
            self.pos = tess_tri(self.pos, 11, 10, 6, self.level_of_tess - 1)

        d_min2 = 1.0e5

        dup_ind = np.array([])

        for i in range(self.pos.shape[0] - 1):
            for j in range(i + 1, self.pos.shape[0]):

                dr = self.pos[j, :] - self.pos[i, :]
                
                d2 = np.sum(dr * dr)

                if d2 < 1.0e-6:

                    dup_ind = np.append(dup_ind, j)

                elif d2 < d_min2:

                    d_min2 = d2

        d_min = np.sqrt(d_min2)

        self.pos = np.delete(self.pos, dup_ind, axis=0)
        self.n_particles = self.pos.shape[0]

        """
        d_mean = 0.0
        n_bonds = 0.0

        for i in range(self.pos.shape[0] - 1):
            for j in range(i + 1, self.pos.shape[0]):

                dr = self.pos[j, :] - self.pos[i, :]

                dij = np.sqrt(np.sum(dr * dr))

                if dij < 1.5 * d_min:
                    d_mean += np.sqrt(d2)
                    n_bonds += 1.0

        d_mean /= n_bonds
        """

        #self.R = self.d / d_min
        #self.R = self.d / d_mean
        aa = 0.5 * np.sqrt(3.0) * (self.d * self.d)
        self.R = np.sqrt(aa * float(self.n_particles) / (4.0 * np.pi))

        self.normal = self.pos.copy()
        self.pos *= self.R

    def build_local_integration_grid_hexagon(self, n_div):

        s3_3 = np.sqrt(3.0) / 3.0

        e10 = np.array([self.r_c, - self.r_c * s3_3]) / float(n_div)
        e20 = np.array([self.r_c, self.r_c * s3_3]) / float(n_div)

        _quad_point_x = []
        _quad_point_y = []

        d_a = s3_3 * (self.r_c / float(n_div)) ** 2

        for sector in range(6):

            th = float(sector) * np.pi / 3.0

            ct = np.cos(th)
            st = np.sin(th)

            e1 = np.array([e10[0] * ct - e10[1] * st, e10[0] * st + e10[1] * ct])
            e2 = np.array([e20[0] * ct - e20[1] * st, e20[0] * st + e20[1] * ct])

            for i in range(n_div):
                for j in range(n_div - i):

                    r = (1.0 / 3.0 + i) * e1 + (1.0 / 3.0 + j) * e2

                    _quad_point_x.append(r[0])
                    _quad_point_y.append(r[1])

                for j in range(n_div - i - 1):

                    r = (2.0 / 3.0 + i) * e1 + (2.0 / 3.0 + j) * e2

                    _quad_point_x.append(r[0])
                    _quad_point_y.append(r[1])

        self.quad_point_x = np.array(_quad_point_x)
        self.quad_point_y = np.array(_quad_point_y)

        self.quad_weight = np.ones_like(self.quad_point_x) * d_a

    def build_local_integration_grid_circle(self, n_quad_points):
        
        # Guass-Legendre quadrature on the unit disk (by KyoungJoong Kim and ManSuk Song)
        
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

    def calc_surface_stress(self):

        v_factor = 1.0 / (4.0 * np.pi)  # in units of W / alpha2 = area_per_particle * v_p / alpha2
        s_factor = 1.0 / (4.0 * np.pi)  # in units of F_m / alpha2

        n_legendre = 87

        th = np.linspace(0.0, np.pi, self.n_angle_div)

        th2 = th * th

        cos_th = np.cos(th)
        sin_th = np.sin(th)

        beta = self.R2 / (4.0 * self.alpha2)

        v0_r = v_factor * np.exp(-beta * th2)
        #v0_r = v_factor * np.cos (th / 2) ** (8.0 * beta)

        sig0_r = -s_factor * np.exp(-beta * th2)

        leg = []
        leg_poly = []

        for i in range(n_legendre):
            leg_poly.append(ss.legendre(i))

            leg.append(leg_poly[i](cos_th))

        c_v = np.zeros(n_legendre)
        c_sig = np.zeros(n_legendre)

        for i in range(n_legendre):
            c_v[i] = (2.0 * float(i) + 1.0) / 2.0 * np.trapz(v0_r * leg[i] * sin_th, th)
            c_sig[i] = (2.0 * float(i) + 1.0) / 2.0 * np.trapz(sig0_r * leg[i] * sin_th, th)

        k_acc = np.load("/srv/public/Dropbox/FU Berlin/Projects/Hydrodynamics Coupling/vesicle_hydrodynamics/k_acc.npy")

        fc1_v = np.zeros(n_legendre)
        fc2_v = np.zeros(n_legendre)

        fc1_sig = np.zeros(n_legendre)
        fc2_sig = np.zeros(n_legendre)

        for m in range(1, n_legendre):

            k1 = k_acc[m, 0]
            k2 = k_acc[m, 1]

            fm = float(m)
            fm2 = fm * fm

            fac = c_v[m] / ((fm * (fm + 1.0)) * (k1 - k2))

            fc1_v[m] = fac * k2 * self.R ** (2.0 - k1)
            fc2_v[m] = -fac * k1 * self.R ** (2.0 - k2)

            fac = c_sig[m] / ((k1 * k2 * (k1 + k2) - 3.0 * k1 * k2 - 6.0 * fm2 - 6.0 * fm) * (k1 - k2))

            fc1_sig[m] = fac * k2 * self.R ** (3.0 - k1)
            fc2_sig[m] = -fac * k1 * self.R ** (3.0 - k2)

        self.v_r = np.zeros_like(th)
        self.sig_r = np.zeros_like(th)

        for m in range(n_legendre):

            k1 = k_acc[m, 0]
            k2 = k_acc[m, 1]

            mmp = float(m) * float(m + 1)

            r_to_k1 = self.R ** k1
            r_to_k2 = self.R ** k2

            f_v = fc1_v[m] * r_to_k1 + fc2_v[m] * r_to_k2
            rfp_v = fc1_v[m] * k1 * r_to_k1 + fc2_v[m] * k2 * r_to_k2
            r3fppp_v = fc1_v[m] * k1 * (k1 - 1.0) * (k1 - 2.0) * r_to_k1 +\
                fc2_v[m] * k2 * (k2 - 1.0) * (k2 - 2.0) * r_to_k2

            f_sig = fc1_sig[m] * r_to_k1 + fc2_sig[m] * r_to_k2

            pm_cos_th = leg_poly[m](cos_th)

            self.sig_r += (r3fppp_v - 3.0 * mmp * rfp_v + 6.0 * mmp * f_v) / self.R3 * pm_cos_th

            self.v_r -= f_sig * mmp / self.R2 * pm_cos_th

        self.sig_r *= self.area_per_particle * self.eta / self.alpha2
        self.v_r *= 1.0 / (self.area_per_particle * self.eta * self.alpha2)

    def integrate_over_sphere(self, integrand):

        result = np.zeros([self.n_particles, self.n_particles])

        for i in range(self.n_particles):

            for j in range(i, self.n_particles):

                cos_th0 = np.dot(self.normal[i], self.normal[j])
                sin_th0 = np.linalg.norm(np.cross(self.normal[i], self.normal[j]))

                th0 = np.arctan2(sin_th0, cos_th0)

                integral = 0.0

                for k in range(self.quad_weight.shape[0]):

                    if i == j:
                        d_th = np.sqrt(self.quad_point_x[k] ** 2 + self.quad_point_y[k] ** 2) / self.R
                    else:
                        d_th = self.quad_point_x[k] / self.R
                        #d_phi = (self.quad_point_y[k] / self.R) / sin_th0

                    th = th0 + d_th

                    if th > np.pi:
                        th = 2.0 * np.pi - th

                    f_ind = th / np.pi * float(self.n_angle_div)
                    ind = np.floor(f_ind)
                    frac = f_ind - ind

                    if ind < self.n_angle_div:

                        value = integrand[int(ind)] * (1.0 - frac) + integrand[int(ind) + 1] * frac

                    else:

                        value = 0.0

                    #det_Jac = 1.0 / (self.R2 * np.sin(th))

                    integral += value * self.quad_weight[k]

                result[i, j] = integral

                result[j, i] = result[i, j]

        return result

    def calc_friction_tensor(self):

        # Friction coefficient tensor
        
        # this corresponds to a normalized Gaussian used for point velocities
        # it is necessary in the sense that the "momentum" of the distributed velocity be equal
        # to the actual momentum of the particle

        self.zeta = self.integrate_over_sphere(-self.sig_r)

    def calc_diffusion_tensor_direct(self):

        # Diffusion coefficient tensor

        # this corresponds to a normalized Gaussian used for point velocities
        # it is necessary in the sense that the "momentum" of the distributed velocity be equal
        # to the actual momentum of the particle

        self.D = self.integrate_over_sphere(self.kT * self.v_r)

    def calc_diffusion_tensor_from_friction(self):
        
        if self.zeta.any() is None:
            self.calc_friction_tensor()

        self.D = self.kT * np.linalg.inv(self.zeta)

    # Calculates the derivative of the diffusion tensor with respect to the position of the particle with index "ind".
    def calc_diffusion_tensor_derivative(self, ind):

        eps = 1.0e-6

        self.calc_diffusion_tensor_direct()
        di_0 = np.copy(self.D)
        pos_0 = np.copy(self.pos[ind, :])

        self.pos[ind, 0] += eps * self.d

        self.calc_diffusion_tensor_direct()

        di_x1 = np.copy(self.D)

        self.pos[ind, :] = pos_0[:]

        self.pos[ind, 1] += eps * self.d

        self.calc_diffusion_tensor_direct()

        di_y1 = np.copy(self.D)

        d_di_d_x = (di_x1 - di_0) / eps
        d_di_d_y = (di_y1 - di_0) / eps

        self.pos[ind, :] = pos_0[:]

        self.D = di_0

        return d_di_d_x, d_di_d_y

    def calc_rnd_decomposed_tensor(self):

        if self.D.any() is None:
            self.calc_diffusion_tensor_direct()

        self.A = np.linalg.cholesky(self.D)

    def calc_vesicle_diffusion_coeff(self):

        if self.D.any() is None:
            self.calc_diffusion_tensor_direct()

        var = 0.0

        D_para = self.kT / (6.0 * np.pi * 0.5 * self.d)

        for i in range(self.n_particles):

            cos_th_i = self.normal[i][2]
            sin_th_i = np.sqrt(1.0 - cos_th_i * cos_th_i)

            var += sin_th_i * sin_th_i * (2.0 * D_para)

            for j in range(self.n_particles):

                cos_th_j = self.normal[j][2]
                sin_th_j = np.sqrt(1.0 - cos_th_j * cos_th_j)

                # var += 2.0 * self.D[i, j] * np.dot (self.normal[i], self.normal[j])
                var += cos_th_i * cos_th_j * (2.0 * self.D[i, j])

        var /= float(self.n_particles) ** 2

        return var / 3.0

    def get_histogram(self, q, _n_bins):

        # Histogram of a matrix q based on inter-particle distances
        
        r = np.linspace(0.0, np.pi * self.R, _n_bins)

        dr = r[1] - r[0]

        n_hist = np.zeros(_n_bins)

        q_hist = np.zeros(_n_bins)
        q_min_hist = np.ones(_n_bins) * 1.0e10
        q_max_hist = np.ones(_n_bins) * -1.0e10

        for i in range(self.n_particles):
            
            for j in range(i, self.n_particles):
                
                cos_th = np.dot(self.normal[i], self.normal[j])
                sin_th = np.linalg.norm(np.cross(self.normal[i], self.normal[j]))

                th0 = np.arctan2(sin_th, cos_th)

                r_ij = self.R * th0

                ind = np.floor(r_ij / dr).astype('int')
                
                q_hist[ind] += q[i, j]

                if q[i, j] < q_min_hist[ind]:
                    q_min_hist[ind] = q[i, j]

                if q[i, j] > q_max_hist[ind]:
                    q_max_hist[ind] = q[i, j]

                n_hist[ind] += 1.0

        nz_ind = np.nonzero(n_hist)
        
        q_mean_hist = q_hist[nz_ind] / n_hist[nz_ind]

        return r[nz_ind], q_mean_hist, q_min_hist[nz_ind], q_max_hist[nz_ind]

    def find_neighbors(self, ind0):

        nei_ind = []

        for i in range(self.n_particles):

            cos_th = np.dot(self.normal[i], self.normal[ind0])
            sin_th = np.linalg.norm(np.cross(self.normal[i], self.normal[ind0]))

            th0 = np.arctan2(sin_th, cos_th)

            r_ij = self.R * th0

            if np.abs(r_ij - self.d) < 1.0e-6 * self.d:

                nei_ind.append(i)

        if len(nei_ind) == 0:
            raise ValueError("No neighbors found!")

        return np.array(nei_ind)
