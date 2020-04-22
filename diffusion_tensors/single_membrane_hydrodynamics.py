import numpy as np
import scipy.special as ss
import surface_integration as surf

class SingleMembraneHydro:
    
    def __init__ (self, _lattice_parameter,
                  _lattice_n_x, _lattice_n_y,
                  _temperature, _viscosity, _alpha_over_a,
                  _n_quadrature_points=4):
        
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

        self.__build_plane ()

        self.r_c = np.sqrt(self.area_per_particle / np.pi)

        print(f"cut-off radius = {self.r_c}")

        self.quad_point_x, self.quad_point_y, self.quad_weight =\
            surf.build_local_integration_grid_circle (_n_quadrature_points, self.r_c)

        self.zeta = None
        self.D = None
        self.A = None
        
    def __build_plane (self):
        
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

    def __wrap (self, dx, dy):

        wdx = dx - np.floor(dx / self.L_x + 0.5) * self.L_x
        wdy = dy - np.floor(dy / self.L_y + 0.5) * self.L_y

        return wdx, wdy

    def calc_friction_tensor (self):
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
                    
                    dx, dy = self.__wrap(dx, dy)

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

    def calc_diffusion_tensor_direct(self, cut_off=np.inf):
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

                    dx, dy = self.__wrap(dx, dy)

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

    def calc_diffusion_tensor_from_inverse(self):

        self.calc_friction_tensor()

        self.D = self.kT * np.linalg.inv(self.zeta)

    def get_histogram(self, M, _n_bins):

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

                dx, dy = self.__wrap (dx, dy)

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
