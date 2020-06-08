import argparse
import numpy as np
import MDAnalysis as md
from propensity import Propensity
from order_parameter import OrderParameter

parser = argparse.ArgumentParser()
parser.add_argument('--s', type=str, help='topology file')
parser.add_argument('--f', type=str, help='trajectory file')
parser.add_argument('--o', type=str, help='output file')
parser.add_argument('--t', type=float, help='propensity(t)')
parser.add_argument('--n_data_frame', type=int, help='number of frames we use for feature vector extraction')
parser.add_argument('--n_neighbor', type=int, default=8, help='number of neighbours')

args = parser.parse_args()
params = vars(args)

u = md.Universe(params['s'], params['f'])

prop = Propensity(u, t=params['t'], n_data_frame=params['n_data_frame'])
propensity = prop.get_propensity()
print(prop.frame_list)

op = OrderParameter(u, frame_list=prop.frame_list)
op_list = op.get_order_parameter()

np.savez(params['o'], feature=op_list, propensity=propensity)