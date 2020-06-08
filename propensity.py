import argparse
import numpy as np
import MDAnalysis as md
import tqdm



class Propensity:
    def __init__(self, universe, t, n_data_frame):
        self.universe = universe
        self.t = t  # ps
        self.n_data_frame = n_data_frame

        self.n_frame = len(self.universe.trajectory)
        self.dt = self.universe.trajectory[1].time - self.universe.trajectory[0].time
        self.n_frame_per_t = int(self.t/self.dt)+1

        if self.n_frame < (self.n_frame_per_t * self.n_data_frame):
            print("Total frame in trajectory file: {%d}".format(self.n_frame))
            print("Frame for one dataset: {%d}".format(self.n_frame_per_t))
            print("Number of dataset: {%d}".format(self.n_data_frame))
            print("Please make new numbers.")
            exit(1)

        self.ow = self.universe.select_atoms("name OW")
        self.n_ow = len(self.ow)

        self.frame_list = []
        for i in range(self.n_data_frame):
            self.frame_list.append(i*(self.n_frame_per_t))


    def _relocate_position(self):
        box_vec = self.universe.trajectory[0].dimensions[:3]     # only NVT simulation
        n_period_mat = np.zeros((self.n_ow, 3)) 

        pos_mat3 = []
        for i in tqdm.tqdm(range(self.n_frame-1)):
            ts = self.universe.trajectory[i]
            pos_bef = self.ow.positions

            ts = self.universe.trajectory[i+1]
            pos_aft = self.ow.positions

            box_mat = np.tile(box_vec, (self.n_ow, 1))
            pos_bef += np.multiply(n_period_mat, box_mat)
            pos_aft += np.multiply(n_period_mat, box_mat)

            pbc_pos_aft = np.copy(pos_aft)
            for j in range(3):
                mask1 = pos_aft[:,j] - pos_bef[:,j] > box_vec[j]/2
                mask2 = pos_bef[:,j] - pos_aft[:,j] > box_vec[j]/2
                pbc_pos_aft[mask1,j] -= box_vec[j]
                pbc_pos_aft[mask2,j] += box_vec[j]
                n_period_mat[mask1,j] -= 1
                n_period_mat[mask2,j] += 1

            if i == 0:
                pos_mat3.append(pos_bef)
            pos_mat3.append(pbc_pos_aft)

        pos_mat3 = np.array(pos_mat3)
        return(pos_mat3)


    def get_propensity(self):
        propensity_mat = []

        pos_mat3 = self._relocate_position()
        for frame in tqdm.tqdm(self.frame_list, total=len(self.frame_list)):
            pos_mat_i = pos_mat3[frame]
            pos_mat_j = pos_mat3[frame+self.n_frame_per_t]
            propensity_vec = self._distance(pos_mat_i, pos_mat_j)
            propensity_mat.append(propensity_vec)
        
        propensity_mat = np.array(propensity_mat)
        return(propensity_mat)

    def _distance(self, x_i, x_j):
        dist = np.zeros(len(x_i))
        for i in range(3):
            dist += (x_j[:,i] - x_i[:,i])**2
        
        return(np.sqrt(dist))



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    parser = argparse.ArgumentParser()
    parser.add_argument('--s', type=str, help='topology file')
    parser.add_argument('--f', type=str, help='trajectory file')
    parser.add_argument('--o', type=str, help='output file')
    parser.add_argument('--n_neighbor', type=int, default=8, help='number of neighbours')

    args = parser.parse_args()
    params = vars(args)

    u = md.Universe(params['s'], params['f'])
    prop = Propensity(u, t=5, n_data_frame=100)
    propensity = prop.get_propensity()

    sns.distplot(propensity.flatten())
    plt.xlim((0, 10))
    plt.show()




