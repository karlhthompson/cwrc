from __future__ import absolute_import, print_function

import argparse
import json
import pprint
import os
import os.path

from gym import spaces
import h5py
import uuid
import numpy as np
import tensorflow as tf

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

import rltools.rltools.algos
import rltools.rltools.log
import rltools.rltools.util
import rltools.rltools.samplers
from runners.run_contworld import ObservationBuffer
from envs import MAContWorld
from rltools.rltools.baselines.linear import LinearFeatureBaseline
from rltools.rltools.baselines.mlp import MLPBaseline
from rltools.rltools.baselines.zero import ZeroBaseline
from rltools.rltools.policy.gaussian import GaussianMLPPolicy

from visualization import Evaluator, Visualizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='./visualization/data/itr_2322.pkl')  # defaultIS.h5/snapshots/iter0000480
    parser.add_argument('--vid', type=str, default='./visualization/videos/contworldvid.mp4')
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--heuristic', action='store_true', default=False)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--n_trajs', type=int, default=48)
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--same_con_pol', action='store_true', default=False)
    args = parser.parse_args()

    with open('./visualization/data/params.json', 'r') as df:
        train_args = json.load(df)

    env = MAContWorld(n_rovers=train_args['n_rovers'],
                      n_areas_of_int=train_args['n_areas_of_int'],
                      n_coop=train_args['n_coop'],
                      n_crater=train_args['n_crater'],
                      n_sensors=train_args['n_sensors'],
                      scout_reward=train_args['scout_reward'],
                      crater_reward=train_args['crater_reward'],
                      control_penalty=-.5,
                      reward_mech='Local',
                      encounter_reward=train_args['encounter_reward'],  #train_args['encounter_reward'],
                      n_obstacles=3,
                      addid=True,
                      _speed_features=True,
                      obstacle_loc=None)

    env.reset()

    if train_args['buffer_size'] > 1:
        env = ObservationBuffer(env, train_args['buffer_size'])

    hpolicy = None # heuristic policy

    if args.evaluate:
        minion = Evaluator(env, train_args, args.n_steps, args.n_trajs, args.deterministic,
                           'heuristic' if args.heuristic else 'rllab')
        evr = minion(args.filename, same_con_pol=args.same_con_pol,
                     hpolicy=hpolicy)
        from tabulate import tabulate
        print(tabulate(evr, headers='keys'))
    else:
        minion = Visualizer(env, train_args, args.n_steps, args.n_trajs, args.deterministic,
                            'heuristic' if args.heuristic else 'rllab')
        rew, info = minion(args.filename, vid=args.vid, hpolicy=hpolicy)
        pprint.pprint(rew)
        pprint.pprint(info)


if __name__ == '__main__':
    main()
