import argparse
import numpy as np
import MDAnalysis as md
from parameter import OrderParameter, Adjacency

parser = argparse.ArgumentParser()
parser.add_argument('--s', type=str, help='topology file')
parser.add_argument('--f', type=str, help='trajectory file')
parser.add_argument('--o', type=str, help='output file')
parser.add_argument('--t', type=float, help='propensity(t)')
parser.add_argument('--n_data_frame', type=int, help='number of frames we use for feature vector extraction')
parser.add_argument('--n_neighbor', type=int, default=10, help='number of neighbours')

args = parser.parse_args()
params = vars(args)

u = md.Universe(params['s'], params['f'])


par = OrderParameter(u, frame_list=np.linspace(0, 2500, 101).astype(int))
par.get_order_parameter()

adj = Adjacency(u, frame_list=np.linspace(0, 2500, 101).astype(int))
adj.get_adj()

np.savez(params['o'], 
    feature=par.fea_list,
    adj=adj.adj_list,
    q_tet=par.q_tet_list,
    lsi=par.lsi_list,)