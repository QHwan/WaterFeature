import numpy as np
from numba import jit
import scipy

'''
cpdef long[:, :] get_adj(double[:, :] sqr_dist_ow_mat, double r_cut):
    cdef long i, j, n_ow
    cdef long[:] idx_sorted_sqr_dist_ow_vec
    cdef long[:, :] adj
    cdef double[:] sqr_dist_ow_vec, sorted_sqr_dist_ow_vec

    n_ow = len(sqr_dist_ow_mat)
    adj = np.zeros((n_ow, n_ow), dtype=long)

    for i in range(n_ow):
        sqr_dist_ow_vec = sqr_dist_ow_mat[i]

        idx_sorted_sqr_dist_ow_vec = np.argsort(sqr_dist_ow_vec, kind='mergesort')
        sorted_sqr_dist_ow_vec = np.sort(sqr_dist_ow_vec, kind='mergesort')

        for j in range(1, n_ow):
            if sorted_sqr_dist_ow_vec[j] > r_cut*r_cut:
                break
            else:
                #adj[i, idx_sorted_sqr_dist_ow_vec[j]] = 1
                adj[i, idx_sorted_sqr_dist_ow_vec[j]] = 1/np.sqrt(sorted_sqr_dist_ow_vec[j])

    return adj



cpdef double[:, :] get_a_hat(long[:, :] adj):
    cdef long i, j, n_ow
    cdef double[:, :] a_hat, buf

    n_ow = np.shape(adj)[0]
    a_hat = np.zeros((n_ow, n_ow))

    buf = adj + np.eye(n_ow)

    for i in range(n_ow):
        for j in range(n_ow):
            if i == j:
                a_hat[i, j] = 1/(np.sum(buf[i])+1)
            elif buf[i, j] == 0:
                pass
            else:
                a_hat[i, j] = buf[i, j] / sqrt((np.sum(buf[i])+1) * (np.sum(buf[j])+1))

    return a_hat


cpdef double[:, :] get_inv_a_hat(long[:, :] adj):
    cdef long i, j, n_ow
    cdef double[:, :] inv_a_hat, buf

    n_ow = np.shape(adj)[0]
    inv_a_hat = np.zeros((n_ow, n_ow))

    buf = adj + np.eye(n_ow)

    for i in range(n_ow):
        for j in range(n_ow):
            if i == j:
                inv_a_hat[i, j] = 2/(np.sum(buf[i])+2)
            elif buf[i, j] == 0:
                pass
            else:
                inv_a_hat[i, j] = -buf[i, j] / sqrt((np.sum(buf[i])+2) * (np.sum(buf[j])+2))

    return inv_a_hat


'''
@jit(nopython=True)
def get_q_tet(pos_ow_mat, pos_ow_vec, box, idx_sorted_sqr_dist_ow_vec):

    q_tet = 0.

    for i in range(1,4):
        idx_i = idx_sorted_sqr_dist_ow_vec[i]
        pos_i_vec = pos_ow_mat[idx_i]
        pbc_pos_i_vec = np.zeros(3)
        pbc_pos_i_vec = pbc(pos_i_vec, pos_ow_vec, box)

        for j in range(i+1,5):
            idx_j = idx_sorted_sqr_dist_ow_vec[j]
            pos_j_vec = pos_ow_mat[idx_j]
            pbc_pos_j_vec = np.zeros(3)
            pbc_pos_j_vec = pbc(pos_j_vec, pos_ow_vec, box)

            cos_angle = angle(pos_ow_vec, pbc_pos_i_vec, pbc_pos_j_vec)

            q_tet += (cos_angle+1./3.)**2

    q_tet = -0.375*q_tet
    q_tet = 1 + q_tet

    return q_tet


@jit(nopython=True)
def pbc(vec, ref_vec, box_vec):
    o_vec = np.zeros(3)
    
    for i in range(3):
        o_vec[i] = vec[i]
        
        if vec[i] - ref_vec[i] > 0.5*box_vec[i]:
            o_vec[i] -= box_vec[i]
        elif vec[i] - ref_vec[i] < -0.5*box_vec[i]:
            o_vec[i] += box_vec[i]

    return o_vec


@jit(nopython=True)
def angle(vec, vec1, vec2):
    v1 = np.zeros(3)
    v2 = np.zeros(3)

    for i in range(3):
        v1[i] = vec1[i] - vec[i]
        v2[i] = vec2[i] - vec[i]

    norm1 = np.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    norm2 = np.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

    for i in range(3):
        v1[i] /= norm1
        v2[i] /= norm2

    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

@jit(nopython=True)
def cartesian_to_spherical(x, y, z):
    xsq_plus_ysq = x**2 + y**2

    r = np.sqrt(xsq_plus_ysq + z**2)
    theta = np.arctan2(z, np.sqrt(xsq_plus_ysq))
    pi = np.arctan2(y,x)

    return np.array([r, theta, pi])
'''
            


cpdef double[:,:] pbc_mol(double[:,:] mat, double[:] ref_vec, double[:] box_vec):
    cdef int i, j, n_row

    cdef double[:,:] o_mat

    o_mat = np.zeros((3,3), dtype=np.double)
    
    n_row = mat.shape[0] 
    
    for i in range(3):
        if mat[0,i] - ref_vec[i] > 0.5*box_vec[i]:
            for j in range(n_row):
                o_mat[j,i] = mat[j,i] - box_vec[i]
        elif mat[0,i] - ref_vec[i] < -0.5*box_vec[i]:
            for j in range(n_row):
                o_mat[j,i] = mat[j,i] + box_vec[i]
        else:
            for j in range(n_row):
                o_mat[j,i] = mat[j,i]

    return o_mat
            

cpdef double norm(double[:] vec):
    return sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

cpdef double distance(double[:] vec1, double[:] vec2):
    return sqrt((vec2[0]-vec1[0])**2 + (vec2[1]-vec1[1])**2 + (vec2[2]-vec1[2])**2)

cpdef double square_distance(double[:] vec1, double[:] vec2):
    return (vec2[0]-vec1[0])**2 + (vec2[1]-vec1[1])**2 + (vec2[2]-vec1[2])**2

'''



'''
cpdef double[:,:] distance_matrix(double[:,:] position_matrix, double[:] box_vec, double rcut):
    cdef int i, j, k, nrow, ncol
    cdef double[:] vec1, vec2, vec3
    cdef double[:,:] dist_mat

    nrow = position_matrix.shape[0]

    dist_mat = np.zeros((nrow, nrow), dtype=np.double) 

    for i in range(nrow):
        vec1 = position_matrix[i]
        
        for j in range(i+1, nrow):
            vec2 = position_matrix[j]
            vec3 = np.zeros(3, dtype=np.double)
            vec3 = pbc(vec2, vec1, box_vec)
            
            #if abs(vec3[0]-vec1[0])>rcut or abs(vec3[1]-vec1[1])>rcut or abs(vec3[2]-vec1[2])>rcut:
            #    dist_mat[i,j] = rcut
            #else:
            dist_mat[i,j] = distance(vec1, vec3)
            
            dist_mat[j,i] = dist_mat[i,j]

    return dist_mat




'''

