import numpy as np
import scipy.special as ss
import surface_integration as surf

class VesicleHydro:

    def __init__(self, _lattice_parameter, _level_of_tess,
                 _temperature, _viscosity, _alpha_over_a,
                 _n_quadrature_points=4):
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

        self.__build_sphere()

        self.R2 = self.R * self.R
        self.R3 = self.R2 * self.R

        self.area_per_particle = 4.0 * np.pi * self.R ** 2 / float(self.n_particles)

        self.r_c = np.sqrt(self.area_per_particle / np.pi)

        print("Number of particles = ", self.n_particles)
        print("Vesicle radius = ", self.R)
        print("Area per particle = ", self.area_per_particle)
        print("Effective radius of each particle = ", self.r_c)

        self.quad_point_x, self.quad_point_y, self.quad_weight =\
            surf.build_local_integration_grid_circle(_n_quadrature_points, self.r_c)

        self.v_r = None
        self.sig_r = None

        self.__calc_surface_stress()

        self.zeta = None
        self.D = None
        self.A = None

    def __build_sphere(self):

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

    def __calc_surface_stress(self):

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

        k_acc = np.load("../hydrodynamics/k_acc.npy")

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

    def __integrate_over_sphere(self, integrand):

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

        self.zeta = self.__integrate_over_sphere(-self.sig_r)

    def calc_diffusion_tensor_direct(self):

        # Diffusion coefficient tensor

        # this corresponds to a normalized Gaussian used for point velocities
        # it is necessary in the sense that the "momentum" of the distributed velocity be equal
        # to the actual momentum of the particle

        self.D = self.__integrate_over_sphere(self.kT * self.v_r)

    def calc_diffusion_tensor_from_friction(self):
        
        if self.zeta.any() is None:
            self.calc_friction_tensor()

        self.D = self.kT * np.linalg.inv(self.zeta)

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
