import math
import argparse
import pickle

import numpy as np
import scipy
import scipy.sparse as sp
from numba import jit

import MDAnalysis as md
import MDAnalysis.analysis.distances as mdanadist
import tqdm
from util import *

def get_ylm_matrix(spherical_coordinate_matrix, l):
    num_vectors = len(spherical_coordinate_matrix)

    ylm_matrix = np.zeros((num_vectors, l*2+1))

    spherical_coordinate_matrix = np.array(spherical_coordinate_matrix)

    theta_vec = spherical_coordinate_matrix[:,1]
    pi_vec = spherical_coordinate_matrix[:,2]

    for j in range(0,l):
        m = j-l
        ylm_matrix[:,j] = np.sqrt(2)*(-1**m)*(scipy.special.sph_harm(m,l,pi_vec,theta_vec)).imag
    
    for j in range(l,l+1):
        ylm_matrix[:,j] = scipy.special.sph_harm(0,l,pi_vec,theta_vec).real

    for j in range(l+1,2*l+1):
        m = j-l
        ylm_matrix[:,j] = np.sqrt(2)*(-1**m)*(scipy.special.sph_harm(m,l,pi_vec,theta_vec)).real

    return ylm_matrix


@jit()
def ylm_to_qlm(ylm_matrix, l):
    num_vectors = len(ylm_matrix)

    qlm_vector = np.zeros(2*l+1)

    for i in range(num_vectors):
        qlm_vector += ylm_matrix[i]

    qlm_vector /= num_vectors

    return qlm_vector

def qlm_to_qlm_average(qlm_matrix, distance_ow_matrix, n_neighbor, l):
    qlm_average_matrix = []

    num_ow = len(distance_ow_matrix)
    
    for i in range(num_ow):

        distance_ow_vector = distance_ow_matrix[i]
        
        index_sorted_distance_ow_vector = np.argsort(distance_ow_vector,
                                                    kind='mergesort')
        
        qlm_average_vector = np.zeros(2*l+1)        

        qlm_average_vector += qlm_matrix[i]
        
        for j in range(1, n_neighbor+1):
            
            index_j = index_sorted_distance_ow_vector[j]
            
            qlm_average_vector += qlm_matrix[index_j]
            
        qlm_average_vector /= n_neighbor+1
        
        qlm_average_matrix.append(qlm_average_vector)

    return np.array(qlm_average_matrix)
 
@jit()
def qlm_average_to_ql(qlm_average_matrix, l=4):
    ql_vector = []

    num_ow = len(qlm_average_matrix)
    
    for i in range(num_ow):
        
        qlm_average_vector = qlm_average_matrix[i]
        
        ql = 0
        
        for qlm_average in qlm_average_vector:

            ql += qlm_average**2
            
        ql *= 4*np.pi/(2*l+1)
        ql = np.sqrt(ql)
        
        ql_vector.append(ql)

    return ql_vector





class OrderParameter:
    def __init__(self, universe, n_neighbor=8, frame_list=None):
        self.universe = universe
        self.n_frame = len(self.universe.trajectory)
        self.n_neighbor = n_neighbor
        self.ow = self.universe.select_atoms("name OW")
        self.n_ow = len(self.ow)
        self.n_fea = 12
        self.frame_list = frame_list
        if self.frame_list is None:
            self.frame_list = np.linspace(0, self.n_frame-1, self.n_frame).astype(int)

    def get_order_parameter(self):
        n_frame_list = len(self.frame_list)
        feat_list = np.zeros((n_frame_list, self.n_ow, self.n_fea))
        d_list = np.zeros((n_frame_list, self.n_ow, 5))
        q_tet_list = np.zeros((n_frame_list, self.n_ow, 1))
        lsi_list = np.zeros((n_frame_list, self.n_ow, 1))
        q_list = np.zeros((n_frame_list, self.n_ow, 5))

        for i, frame in tqdm.tqdm(enumerate(self.frame_list), total=n_frame_list):
            ts = self.universe.trajectory[frame]
            box = ts.dimensions
            pos_ow_mat = self.ow.positions

            dist_ow_mat = np.zeros((self.n_ow, self.n_ow))
            dist_ow_mat = mdanadist.distance_array(pos_ow_mat,
                            pos_ow_mat,
                            box=box,
                            )

            for j in range(self.n_ow):            
                dist_ow_vec = dist_ow_mat[j]

                idx_sorted_dist_ow_vec = np.argsort(dist_ow_vec, kind='mergesort')
                sorted_dist_ow_vec = np.sort(dist_ow_vec, kind='mergesort')

                pos_ow_vec = pos_ow_mat[j]

                
                # d
                d_vec = np.zeros(5)

                for k in range(len(d_vec)):
                    d_vec[k] = sorted_dist_ow_vec[k+1]

                for k in range(5):
                    d_list[i, j, k] = d_vec[k]
                

                # q_tet
                q_tet = get_q_tet(pos_ow_mat, pos_ow_vec, box, idx_sorted_dist_ow_vec)
                q_tet_list[i, j, 0] = q_tet


                # LSI
                lsi = 0.

                lsi_dist_vec = []
                for k in range(1, self.n_ow):
                    lsi_dist_vec.append(sorted_dist_ow_vec[k])
                    if sorted_dist_ow_vec[k] > 3.7:
                        break

                diff_lsi_dist_vec = []
                for k in range(len(lsi_dist_vec)-1):
                    diff_lsi_dist_vec.append(lsi_dist_vec[k+1] - lsi_dist_vec[k])

                avg_diff = np.mean(diff_lsi_dist_vec)

                for k in range(len(diff_lsi_dist_vec)):
                    lsi += (diff_lsi_dist_vec[k] - avg_diff)**2

                if len(diff_lsi_dist_vec) == 0:
                    lsi = 0
                else:
                    lsi /= len(diff_lsi_dist_vec)

                lsi_list[i, j, 0] = lsi


            # q3 & q4 & q5 & q6 & q12
            q3lm_mat = []
            q4lm_mat = []
            q5lm_mat = []
            q6lm_mat = []
            q12lm_mat = []

            for j in range(self.n_ow):

                dist_ow_vec = dist_ow_mat[j]

                sorted_dist_ow_vec = np.sort(dist_ow_vec, kind='mergesort')
                idx_sorted_dist_ow_vec = np.argsort(dist_ow_vec, kind='mergesort')

                angle_matrix = []

                pos_ow_vec = pos_ow_mat[j]
                
                sph_coord_mat = []


                for k in range(1, self.n_neighbor+1):

                    idx_k = idx_sorted_dist_ow_vec[k]

                    pos_ow_i_vec = pos_ow_mat[idx_k]
                    pos_ow_i_vec = pbc(pos_ow_i_vec, pos_ow_vec, box)

                    x, y, z = pos_ow_i_vec - pos_ow_vec

                    r, theta, pi = cartesian_to_spherical(x, y, z)

                    sph_coord_mat.append([r, theta, pi])

                sph_coord_mat = np.array(sph_coord_mat)

                y3lm_mat = get_ylm_matrix(sph_coord_mat, l=3)
                y4lm_mat = get_ylm_matrix(sph_coord_mat, l=4)
                y5lm_mat = get_ylm_matrix(sph_coord_mat, l=5)
                y6lm_mat = get_ylm_matrix(sph_coord_mat, l=6)
                y12lm_mat = get_ylm_matrix(sph_coord_mat, l=12)

                q3lm_vec = ylm_to_qlm(y3lm_mat, l=3)
                q4lm_vec = ylm_to_qlm(y4lm_mat, l=4)
                q5lm_vec = ylm_to_qlm(y5lm_mat, l=5)
                q6lm_vec = ylm_to_qlm(y6lm_mat, l=6)
                q12lm_vec = ylm_to_qlm(y12lm_mat, l=12)

                q3lm_mat.append(q3lm_vec)
                q4lm_mat.append(q4lm_vec)
                q5lm_mat.append(q5lm_vec)
                q6lm_mat.append(q6lm_vec)
                q12lm_mat.append(q12lm_vec)

            q3lm_avg_mat = qlm_to_qlm_average(q3lm_mat, dist_ow_mat, self.n_neighbor, l=3)
            q4lm_avg_mat = qlm_to_qlm_average(q4lm_mat, dist_ow_mat, self.n_neighbor, l=4)
            q5lm_avg_mat = qlm_to_qlm_average(q5lm_mat, dist_ow_mat, self.n_neighbor, l=5)
            q6lm_avg_mat = qlm_to_qlm_average(q6lm_mat, dist_ow_mat, self.n_neighbor, l=6)
            q12lm_avg_mat = qlm_to_qlm_average(q12lm_mat, dist_ow_mat, self.n_neighbor, l=12)

            q3 = qlm_average_to_ql(q3lm_avg_mat, l=3)
            q4 = qlm_average_to_ql(q4lm_avg_mat, l=4)
            q5 = qlm_average_to_ql(q5lm_avg_mat, l=5)
            q6 = qlm_average_to_ql(q6lm_avg_mat, l=6)
            q12 = qlm_average_to_ql(q12lm_avg_mat, l=12)

            for j in range(self.n_ow):
                q_list[i, j, 0] = q3[j]
                q_list[i, j, 1] = q4[j]
                q_list[i, j, 2] = q5[j]
                q_list[i, j, 3] = q6[j]
                q_list[i, j, 4] = q12[j]

        feat_list = np.concatenate((d_list, q_tet_list, lsi_list, q_list), axis=2)
        return(feat_list)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', type=str, help='topology file')
    parser.add_argument('--f', type=str, help='trajectory file')
    parser.add_argument('--o', type=str, help='output file')
    parser.add_argument('--n_neighbor', type=int, default=8, help='number of neighbours')

    args = parser.parse_args()
    params = vars(args)

    u = md.Universe(params['s'], params['f'])
    op = OrderParameter(u)
    d_list = op.get_order_parameter()

