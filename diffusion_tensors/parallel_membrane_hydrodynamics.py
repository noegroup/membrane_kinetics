import numpy as np
import scipy.special as ss
import scipy.integrate as si
import progressbar

class ParallelMembraneHydro:
    
    def __init__(self, _lattice_parameter,
                 _lattice_n_x, _lattice_n_y, _planes_separation,
                 _temperature, _viscosity, _alpha_over_a,
                 _quadrature_area="circle", _n_quadrature_points=4):
        
        self.d = _lattice_parameter
        self.h = _planes_separation

        self.area_per_particle = 0.5 * np.sqrt(3.0) * (self.d * self.d)
        
        self.n_x = _lattice_n_x
        self.n_y = _lattice_n_y

        self.alpha = _alpha_over_a * self.d
        self.alpha2 = self.alpha * self.alpha
        self.alpha3 = self.alpha2 * self.alpha

        self.eta = _viscosity
        self.T = _temperature

        self.kT = 1.38064852e-2 * self.T

        self.n_particles_in_leaflet = 0
        self.n_particles = 0
        self.pos = None
        self.box_size = np.zeros(3)

        self.build_planes()

        self.quad_point_x = None
        self.quad_point_y = None
        self.quad_weight = None

        if _quadrature_area == "circle":
            
            self.r_c = np.sqrt(self.area_per_particle / np.pi)
            # self.r_c = 0.5 * self.d

            print("Cut-off radius = ", self.r_c)

            self.build_local_integration_grid_circle(_n_quadrature_points)

        elif _quadrature_area == "hexagon":
            
            self.r_c = 0.5 * self.d
            self.build_local_integration_grid_hexagon(_n_quadrature_points)

        else:

            raise ValueError("The integration area '" + _quadrature_area + "' is undefined!")

        self.tab_r = None
        self.tab_dr = None
        self.tab_sig_z_0 = None
        self.tab_sig_z_h = None
        self.tab_v_z_0 = None
        self.tab_v_z_h = None

        self.calc_tabular_functions()

        self.zeta_00 = None
        self.zeta_01 = None
        self.zeta = None
        self.D_00 = None
        self.D_01 = None
        self.D = None
        self.A = None
        
    def build_planes(self):
        
        self.n_particles_in_leaflet = self.n_x * self.n_y
        self.n_particles = 2 * self.n_particles_in_leaflet

        self.pos = np.zeros((self.n_particles, 3))
        
        _l = self.d * 0.5 * np.sqrt(3.0)
        
        for j in range(self.n_y):
            for i in range(self.n_x):
                
                ind_lower = i + j * self.n_x
                ind_upper = ind_lower + self.n_particles_in_leaflet

                fi = float(i)
                fj = float(j)
                fj_mod_2 = float(j % 2)
                
                self.pos[ind_lower, 0] = fi * self.d + 0.5 * fj_mod_2 * self.d
                self.pos[ind_lower, 1] = fj * _l
                self.pos[ind_lower, 2] = 0.0

                self.pos[ind_upper, 0] = self.pos[ind_lower, 0]
                self.pos[ind_upper, 1] = self.pos[ind_lower, 1]
                self.pos[ind_upper, 2] = self.h

        self.box_size[0] = float(self.n_x) * self.d
        self.box_size[1] = float(self.n_y) * _l
        self.box_size[2] = max(self.box_size[0], self.box_size[1])

    def build_local_integration_grid_hexagon(self, n_div):

        s3_3 = np.sqrt(3.0) / 3.0

        e10 = np.array([self.r_c, - self.r_c * s3_3]) / float(n_div)
        e20 = np.array([self.r_c, self.r_c * s3_3]) / float(n_div)

        _quad_point_x = []
        _quad_point_y = []

        d_area = s3_3 * (self.r_c / float(n_div)) ** 2

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

        self.quad_weight = np.ones_like(self.quad_point_x) * d_area

    def build_local_integration_grid_circle(self, n_quad_points):
        
        # Gauss-Legendre quadrature on the unit disk (by KyoungJoong Kim and ManSuk Song)
        
        if n_quad_points == 1:

            w_1 = 3.141592653589793
            x_1 = 0.0
            
            self.quad_point_x = np.array([x_1]) * self.r_c

            self.quad_point_y = np.array([x_1]) * self.r_c
                                            
            self.quad_weight = np.array([w_1]) * self.r_c * self.r_c

        elif n_quad_points == 4:

            w_1 = 0.785398163397448
            x_1 = 0.5
            
            self.quad_point_x = np.array([x_1, -x_1, -x_1, x_1]) * self.r_c

            self.quad_point_y = np.array([x_1, x_1, -x_1, -x_1]) * self.r_c
                                            
            self.quad_weight = np.array([w_1, w_1, w_1, w_1]) * self.r_c * self.r_c

        elif n_quad_points == 8:

            w_1 = 0.732786462492640
            w_2 = 0.052611700904808
            x_1 = 0.650115167343736
            x_2 = 0.888073833977115
            
            self.quad_point_x = np.array([x_1, 0.0, -x_1, 0.0, x_2, -x_2, -x_2, x_2]) * self.r_c

            self.quad_point_y = np.array([0.0, x_1, 0.0, -x_1, x_2, x_2, -x_2, -x_2]) * self.r_c
                                            
            self.quad_weight = np.array([w_1, w_1, w_1, w_1, w_2, w_2, w_2, w_2]) * self.r_c * self.r_c
            
        elif n_quad_points == 12:

            w_1 = 0.232710566932577
            w_2 = 0.387077796006226
            w_3 = 0.165609800458645
            
            x_1 = 0.866025403784439
            x_2 = 0.322914992067400
            x_3 = 0.644171310389465
            
            self.quad_point_x = np.array([x_1, 0.0, -x_1, 0.0, x_2, -x_2, -x_2, x_2, x_3, -x_3, -x_3, x_3]) * self.r_c

            self.quad_point_y = np.array([0.0, x_1, 0.0, -x_1, x_2, x_2, -x_2, -x_2, x_3, x_3, -x_3, -x_3]) * self.r_c
                                            
            self.quad_weight = np.array([w_1, w_1, w_1, w_1, w_2, w_2, w_2, w_2, w_3, w_3, w_3, w_3]) *\
                self.r_c * self.r_c

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

            raise ValueError("No set of points/weights for the choice of " + str(n_quad_points) + " quadrature point!")

    def calc_tabular_functions(self):

        def sig_tilde_z_0(q, _h, _alpha2):

            chi = 2.0 * q * _h

            chi2 = chi * chi
            q2 = q * q
            e_chi = np.exp(-chi)
            e_2chi = e_chi * e_chi

            f = (-e_2chi + 2.0 * chi * e_chi + 1.0) / (e_2chi - (chi2 + 2.0) * e_chi + 1.0)

            return q * np.exp(-_alpha2 * q2) * f

        def sig_tilde_z_h(q, _h, _alpha2):

            chi = 2.0 * q * _h

            chi2 = chi * chi
            q2 = q * q
            e_chi_2 = np.exp(-0.5 * chi)
            e_chi = e_chi_2 * e_chi_2
            e_2chi = e_chi * e_chi

            f = e_chi_2 * ((chi - 2.0) * e_chi + chi + 2.0) / (e_2chi - (chi2 + 2.0) * e_chi + 1.0)

            return q * np.exp(-_alpha2 * q2) * f

        def v_tilde_z_0(q, _h, _alpha2):

            chi = 2.0 * q * _h
            q2 = q * q

            e_chi = np.exp(-chi)
            e_2chi = e_chi * e_chi

            f = (-e_2chi + 2.0 * chi * e_chi + 1.0) / (e_2chi - 2.0 * e_chi + 1.0)

            return np.exp(-_alpha2 * q2) * f / q

        def v_tilde_z_h(q, _h, _alpha2):

            chi = 2.0 * q * _h
            q2 = q * q

            e_chi_2 = np.exp(-0.5 * chi)
            e_chi = e_chi_2 * e_chi_2
            e_2chi = e_chi * e_chi

            f = e_chi_2 * ((chi - 2.0) * e_chi + chi + 2.0) / (e_2chi - 2.0 * e_chi + 1.0)

            return np.exp(-_alpha2 * q2) * f / q

        def hankel_inv(fun, _h, _alpha2,  _r):

            def integrand(q):

                return fun(q, _h, _alpha2) * ss.jv(0, q * _r) * q

            #infinity = 100.0
            infinity = 10.0

            chunk_interval = 0.01 / np.sqrt(_alpha2)
            n_chunks = int(np.ceil(infinity / chunk_interval))

            #_n_gauss = 10
            _n_gauss = 5

            result = 0.0

            for m in range(n_chunks):
                dres, _ = si.fixed_quad(func=integrand,
                                        a=float(m) * chunk_interval, b=float(m + 1) * chunk_interval,
                                        n=_n_gauss)
                result += dres

            return result

        factor_sig = 1.0 / (2.0 * np.pi)
        factor_v = 1.0 / (2.0 * np.pi)

        max_r = 0.5 * np.sqrt(self.box_size[0] ** 2 + self.box_size[1] ** 2)
        _inf = 10000.0

        #self.tab_r = np.linspace(0.0, max_r, 2000)
        self.tab_r = np.linspace(0.0, max_r, 500)
        self.tab_dr = self.tab_r[1] - self.tab_r[0]

        ii_s0 = []
        ii_sh = []
        ii_v0 = []
        ii_vh = []

        print ("preparing inverse Hankel transforms...")

        bar = progressbar.ProgressBar(max_value=len(self.tab_r))
        ind = 0

        for _r in self.tab_r:

            ii_s0.append(hankel_inv(sig_tilde_z_0, self.h, self.alpha2, _r))
            ii_sh.append(hankel_inv(sig_tilde_z_h, self.h, self.alpha2, _r))
            ii_v0.append(hankel_inv(v_tilde_z_0, self.h, self.alpha2, _r))
            ii_vh.append(hankel_inv(v_tilde_z_h, self.h, self.alpha2, _r))

            ind += 1
            bar.update (ind)


        print ("preparing far field...")

        sz0_inf = -2.0 * self.eta * factor_v * hankel_inv(sig_tilde_z_0, self.h, self.alpha2, _inf)
        szh_inf = -2.0 * self.eta * factor_v * hankel_inv(sig_tilde_z_h, self.h, self.alpha2, _inf)

        self.tab_sig_z_0 = -2.0 * self.eta * factor_v * np.array(ii_s0) - sz0_inf
        self.tab_sig_z_h = -2.0 * self.eta * factor_v * np.array(ii_sh) - szh_inf

        vz0_inf = -1.0 / (2.0 * self.eta) * (-factor_sig) * hankel_inv(v_tilde_z_0, self.h, self.alpha2, _inf)
        vzh_inf = -1.0 / (2.0 * self.eta) * (-factor_sig) * hankel_inv(v_tilde_z_h, self.h, self.alpha2, _inf)

        self.tab_v_z_0 = -1.0 / (2.0 * self.eta) * (-factor_sig) * np.array(ii_v0) - vz0_inf
        self.tab_v_z_h = -1.0 / (2.0 * self.eta) * (-factor_sig) * np.array(ii_vh) - vzh_inf

    def wrap(self, dx, dy):

        wdx = dx - np.rint(dx / self.box_size[0]) * self.box_size[0]
        wdy = dy - np.rint(dy / self.box_size[1]) * self.box_size[1]

        return wdx, wdy

    def integrate_over_planes(self, prefactor, func_00, func_01):

        result_00 = np.zeros((self.n_particles_in_leaflet, self.n_particles_in_leaflet))
        result_01 = np.zeros((self.n_particles_in_leaflet, self.n_particles_in_leaflet))

        for i in range(self.n_particles_in_leaflet):

            for j in range(i, self.n_particles_in_leaflet):

                integral_00 = 0.0
                integral_01 = 0.0

                for k in range(self.quad_weight.shape[0]):

                    dx = self.pos[j, 0] + self.quad_point_x[k] - self.pos[i, 0]
                    dy = self.pos[j, 1] + self.quad_point_y[k] - self.pos[i, 1]

                    dx, dy = self.wrap(dx, dy)

                    r_ij = np.sqrt(dx * dx + dy * dy)

                    ind = int(np.floor(r_ij / self.tab_dr))
                    frac = np.floor(r_ij / self.tab_dr) - float(ind)

                    integrand_00 = func_00[ind] * (1.0 - frac) + func_00[ind + 1] * frac
                    integrand_01 = func_01[ind] * (1.0 - frac) + func_01[ind + 1] * frac

                    integral_00 += integrand_00 * self.quad_weight[k]
                    integral_01 += integrand_01 * self.quad_weight[k]

                result_00[i, j] = prefactor * integral_00
                result_00[j, i] = result_00[i, j]

                result_01[i, j] = prefactor * integral_01
                result_01[j, i] = result_01[i, j]

        result = np.block([[result_00, result_01],
                           [result_01, result_00]])

        return result_00, result_01, result

    def calc_friction_tensor(self):

        # Friction coefficient tensor

        # this corresponds to a normalized Gaussian used for point velocities
        # it is necessary in the sense that the "momentum" of the distributed velocity be equal
        # to the actual momentum of the particle

        factor = self.area_per_particle

        self.zeta_00, self.zeta_01, self.zeta = self.integrate_over_planes(factor, -self.tab_sig_z_0, -self.tab_sig_z_h)

    def calc_diffusion_tensor_direct(self, cut_off=np.inf):

        # Diffusion coefficient tensor

        # this corresponds to a normalized Gaussian used for point velocities
        # it is necessary in the sense that the "momentum" of the distributed velocity be equal
        # to the actual momentum of the particle

        # The surface integral of velocity distribution is W, which is A_p * v_p

        factor = self.kT / self.area_per_particle

        self.D_00, self.D_01, self.D = self.integrate_over_planes(factor, self.tab_v_z_0, self.tab_v_z_h)

    def calc_diffusion_tensor_from_inverse(self):
        
        if self.zeta.any() is None:
            self.calc_friction_tensor()

        self.D = self.kT * np.linalg.inv(self.zeta)

    def calc_diffusion_tensor_derivative(self, ind, cut_off=np.inf):

        # Calculates the derivative of the diffusion tensor with
        #  respect to the position of the particle with index "ind".

        eps = 1.0e-6

        self.calc_diffusion_tensor_direct(cut_off)
        D0 = np.copy (self.D)
        pos0 = np.copy (self.pos[ind, :])

        self.pos[ind, 0] += eps * self.d

        self.calc_diffusion_tensor_direct(cut_off)

        Dx1 = np.copy (self.D)

        self.pos[ind, :] = pos0[:]

        self.pos[ind, 1] += eps * self.d

        self.calc_diffusion_tensor_direct(cut_off)

        Dy1 = np.copy (self.D)

        dD_dx = (Dx1 - D0) / eps
        dD_dy = (Dy1 - D0) / eps

        self.pos[ind, :] = pos0[:]

        self.D = D0

        return dD_dx, dD_dy

    def calc_cholesky_decomposed_tensor(self, cut_off=np.inf):

        if self.D.any() is None:
            self.calc_diffusion_tensor_direct(cut_off)

        self.A = np.linalg.cholesky(self.D)

    def get_histogram(self, mat, _n_bins):

        # Histogram of a matrix mat based on inter-particle distances

        hist_r = np.linspace(0.0, np.sqrt(np.sum(self.box_size ** 2)), _n_bins)

        hist_dr = hist_r[1] - hist_r[0]

        n_hist = np.zeros (_n_bins)

        mat_hist = np.zeros(_n_bins)
        mat_min_hist = np.ones(_n_bins) * 1.0e10
        mat_max_hist = np.ones(_n_bins) * -1.0e10

        for i in range(mat.shape[0]):

            for j in range(i, mat.shape[1]):

                dx = self.pos[j, 0] - self.pos[i, 0]
                dy = self.pos[j, 1] - self.pos[i, 1]

                dx, dy = self.wrap(dx, dy)

                r_ij = np.sqrt(dx * dx + dy * dy)

                ind = np.round(r_ij / hist_dr).astype(np.int)

                mat_hist[ind] += mat[i, j]

                if mat[i, j] < mat_min_hist[ind]:
                    mat_min_hist[ind] = mat[i, j]

                if mat[i, j] > mat_max_hist[ind]:
                    mat_max_hist[ind] = mat[i, j]

                n_hist[ind] += 1.0

        nz_ind = np.nonzero (n_hist)

        mat_mean_hist = mat_hist[nz_ind] / n_hist[nz_ind]

        return hist_r[nz_ind], mat_mean_hist, mat_min_hist[nz_ind], mat_max_hist[nz_ind]

