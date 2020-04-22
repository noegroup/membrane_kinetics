import numpy as np
import scipy.special as ss
import scipy.integrate as si
import surface_integration as surf
import progressbar

class ParallelMembraneHydro:
    
    def __init__(self, _lattice_parameter,
                 _lattice_n_x, _lattice_n_y, _planes_separation,
                 _temperature, _viscosity, _alpha_over_a,
                 _n_quadrature_points=4):
        
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

        self.__build_planes()

        self.r_c = np.sqrt(self.area_per_particle / np.pi)
        print(f"cut-off radius = {self.r_c}")

        self.quad_point_x, self.quad_point_y, self.quad_weight =\
            surf.build_local_integration_grid_circle(_n_quadrature_points, self.r_c)

        self.tab_r = None
        self.tab_dr = None
        self.tab_sig_z_0 = None
        self.tab_sig_z_h = None
        self.tab_v_z_0 = None
        self.tab_v_z_h = None

        self.__calc_tabular_functions()

        self.zeta_00 = None
        self.zeta_01 = None
        self.zeta = None
        self.D_00 = None
        self.D_01 = None
        self.D = None
        self.A = None
        
    def __build_planes(self):
        
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

    def __calc_tabular_functions(self):

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

    def __wrap(self, dx, dy):

        wdx = dx - np.rint(dx / self.box_size[0]) * self.box_size[0]
        wdy = dy - np.rint(dy / self.box_size[1]) * self.box_size[1]

        return wdx, wdy

    def __integrate_over_planes(self, prefactor, func_00, func_01):

        result_00 = np.zeros((self.n_particles_in_leaflet, self.n_particles_in_leaflet))
        result_01 = np.zeros((self.n_particles_in_leaflet, self.n_particles_in_leaflet))

        for i in range(self.n_particles_in_leaflet):

            for j in range(i, self.n_particles_in_leaflet):

                integral_00 = 0.0
                integral_01 = 0.0

                for k in range(self.quad_weight.shape[0]):

                    dx = self.pos[j, 0] + self.quad_point_x[k] - self.pos[i, 0]
                    dy = self.pos[j, 1] + self.quad_point_y[k] - self.pos[i, 1]

                    dx, dy = self.__wrap(dx, dy)

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

        self.zeta_00, self.zeta_01, self.zeta = self.__integrate_over_planes(factor, -self.tab_sig_z_0, -self.tab_sig_z_h)

    def calc_diffusion_tensor_direct(self, cut_off=np.inf):

        # Diffusion coefficient tensor

        # this corresponds to a normalized Gaussian used for point velocities
        # it is necessary in the sense that the "momentum" of the distributed velocity be equal
        # to the actual momentum of the particle

        # The surface integral of velocity distribution is W, which is A_p * v_p

        factor = self.kT / self.area_per_particle

        self.D_00, self.D_01, self.D = self.__integrate_over_planes(factor, self.tab_v_z_0, self.tab_v_z_h)

    def calc_diffusion_tensor_from_inverse(self):
        
        if self.zeta.any() is None:
            self.calc_friction_tensor()

        self.D = self.kT * np.linalg.inv(self.zeta)

    def get_histogram(self, mat, _n_bins):

        # Histogram of the matrix 'mat' based on inter-particle distances

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

                dx, dy = self.__wrap(dx, dy)

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

