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
    def __init__(self, universe, n_neighbor=10, frame_list=None, angular=False):
        self.universe = universe
        self.n_frame = len(self.universe.trajectory)
        self.n_neighbor = n_neighbor
        self.ow = self.universe.select_atoms("name OW")
        self.n_ow = len(self.ow)
        self.frame_list = frame_list
        if self.frame_list is None:
            self.frame_list = np.linspace(0, self.n_frame-1, self.n_frame).astype(int)

        self.angular = angular
        if self.angular:
            self.n_fea = n_neighbor*4
        else:
            self.n_fea = n_neighbor

    def _normalize(self, _mat):
        mat = np.zeros((3, 3))
        for i in range(3):
            vec = _mat[i]
            norm = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
            mat[i] += vec/norm
        return(mat)

    def get_order_parameter(self):
        n_frame_list = len(self.frame_list)
        self.fea_list = np.zeros((n_frame_list, self.n_ow, self.n_fea))       
        self.q_tet_list = np.zeros((n_frame_list, self.n_ow, 1))
        self.lsi_list = np.zeros((n_frame_list, self.n_ow, 1))

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
                pos_nei_ow_mat = np.zeros((self.n_neighbor, 3))
                d_ij_vec = np.zeros(self.n_neighbor)

                for k in range(self.n_neighbor):
                    pos_nei_ow_mat[k] += pos_ow_mat[idx_sorted_dist_ow_vec[k+1]]
                    d_ij_vec[k] += sorted_dist_ow_vec[k+1]

                
                _rot_mat = np.zeros((3, 3))
                r_ia_vec = pos_nei_ow_mat[0] - pos_ow_vec
                r_ib_vec = pos_nei_ow_mat[1] - pos_ow_vec
                _rot_mat[0] += r_ia_vec
                _rot_mat[1] += r_ib_vec - np.dot(r_ia_vec, r_ib_vec) * r_ia_vec
                _rot_mat[2] += np.cross(r_ia_vec, r_ib_vec)
                rot_mat = self._normalize(_rot_mat).T
                
                if self.angular:
                    for k in range(self.n_neighbor):
                        r_ij_vec = pos_nei_ow_mat[k] - pos_ow_vec
                        rp_ij_vec = np.matmul(r_ij_vec, rot_mat)
                        self.fea_list[i,j,4*k+0] = 1/d_ij_vec[k]
                        self.fea_list[i,j,4*k+1] = rp_ij_vec[0]/(d_ij_vec[k]**2)
                        self.fea_list[i,j,4*k+2] = rp_ij_vec[1]/(d_ij_vec[k]**2)
                        self.fea_list[i,j,4*k+3] = rp_ij_vec[2]/(d_ij_vec[k]**2)
                else:
                    for k in range(self.n_neighbor):
                        self.fea_list[i,j,k] = 1/d_ij_vec[k]

                
                # q_tet
                q_tet = get_q_tet(pos_ow_mat, pos_ow_vec, box, idx_sorted_dist_ow_vec)
                self.q_tet_list[i, j, 0] = q_tet


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

                self.lsi_list[i, j, 0] = lsi



class Adjacency:
    def __init__(self, universe, r_cut=3.5, frame_list=None):
        self.universe = universe
        self.n_frame = len(self.universe.trajectory)
        self.r_cut = r_cut
        self.ow = self.universe.select_atoms("name OW")
        self.n_ow = len(self.ow)
        self.frame_list = frame_list
        if self.frame_list is None:
            self.frame_list = np.linspace(0, self.n_frame-1, self.n_frame).astype(int)

    def get_adj(self):
        n_frame_list = len(self.frame_list)
        self.adj_list = np.zeros((n_frame_list, self.n_ow, self.n_ow))

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
                dist_ow_vec = dist_ow_mat[i]

                idx_sorted_dist_ow_vec = np.argsort(dist_ow_vec, kind='mergesort')
                sorted_dist_ow_vec = np.sort(dist_ow_vec, kind='mergesort')

                for k in range(1, self.n_ow):
                    if sorted_dist_ow_vec[k] > self.r_cut:
                        break
                    else:
                        self.adj_list[i, j, idx_sorted_dist_ow_vec[k]] = 1




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

