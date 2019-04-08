import numpy as np
import tensorflow as tf
from gym import spaces

import argparse
import sys
import datetime
import dateutil
import dateutil.tz
import uuid
import ast
import yaml
import yaml.constructor
import os
import json
import pprint
import h5py

from collections import OrderedDict
from collections import namedtuple

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

from runners import archs
from envs.contworld import *

def tonamedtuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = tonamedtuple(value)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

def get_arch(name):
    constructor = getattr(archs, name)
    return constructor

def comma_sep_ints(s):
    if s:
        return list(map(int, s.split(",")))
    else:
        return []

class RunnerParser(object):

    DEFAULT_OPTS = [
        ('discount', float, 0.95, ''),
        ('gae_lambda', float, 0.99, ''),
        ('n_iter', int, 500, ''),
    ]

    DEFAULT_POLICY_OPTS = [
        ('control', str, 'decentralized', ''),
        ('recurrent', str, None, ''),
        ('baseline_type', str, 'linear', ''),
    ]

    def __init__(self, env_options, **kwargs):
        self._env_options = env_options
        parser = argparse.ArgumentParser(description='Runner')

        parser.add_argument('mode', help='rllab or rltools')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.mode):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        self._mode = args.mode
        getattr(self, args.mode)(self._env_options, **kwargs)

    def update_argument_parser(self, parser, options, **kwargs):
        kwargs = kwargs.copy()
        for (name, typ, default, desc) in options:
            flag = "--" + name
            if flag in parser._option_string_actions.keys():  #pylint: disable=W0212
                print("warning: already have option %s. skipping" % name)
            else:
                parser.add_argument(flag, type=typ, default=kwargs.pop(name, default), help=desc or
                                    " ")
        if kwargs:
            raise ValueError("options %s ignored" % kwargs)

    def rllab(self, env_options, **kwargs):
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        rand_id = str(uuid.uuid4())[:5]
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
        default_exp_name = 'experiment_%s_%s' % (timestamp, rand_id)

        parser = argparse.ArgumentParser()
        parser.add_argument('--exp_name', type=str, default=default_exp_name)
        self.update_argument_parser(parser, self.DEFAULT_OPTS)
        self.update_argument_parser(parser, self.DEFAULT_POLICY_OPTS)

        parser.add_argument(
            '--algo', type=str, default='tftrpo',
            help='Add tf or th to the algo name to run tensorflow or theano version')

        parser.add_argument('--max_path_length', type=int, default=500)
        parser.add_argument('--batch_size', type=int, default=12000)
        parser.add_argument('--n_parallel', type=int, default=1)
        parser.add_argument('--resume_from', type=str, default=None,
                            help='Name of the pickle file to resume experiment from.')

        parser.add_argument('--epoch_length', type=int, default=1000)
        parser.add_argument('--min_pool_size', type=int, default=10000)
        parser.add_argument('--replay_pool_size', type=int, default=500000)
        parser.add_argument('--eval_samples', type=int, default=50000)
        parser.add_argument('--qfunc_lr', type=float, default=1e-3)
        parser.add_argument('--policy_lr', type=float, default=1e-4)

        parser.add_argument('--feature_net', type=str, default=None)
        parser.add_argument('--feature_output', type=int, default=16)
        parser.add_argument('--feature_hidden', type=comma_sep_ints, default='128,64,32')
        parser.add_argument('--policy_hidden', type=comma_sep_ints, default='32')
        parser.add_argument('--conv', type=str, default='')
        parser.add_argument('--conv_filters', type=comma_sep_ints, default='3,3')
        parser.add_argument('--conv_channels', type=comma_sep_ints, default='4,8')
        parser.add_argument('--conv_strides', type=comma_sep_ints, default='1,1')
        parser.add_argument('--min_std', type=float, default=1e-6)
        parser.add_argument('--exp_strategy', type=str, default='ou')
        parser.add_argument('--exp_noise', type=float, default=0.3)

        parser.add_argument('--step_size', type=float, default=0.01, help='max kl wall limit')

        parser.add_argument('--log_dir', type=str, required=False)
        parser.add_argument('--tabular_log_file', type=str, default='progress.csv',
                            help='Name of the tabular log file (in csv).')
        parser.add_argument('--text_log_file', type=str, default='debug.log',
                            help='Name of the text log file (in pure text).')
        parser.add_argument('--params_log_file', type=str, default='params.json',
                            help='Name of the parameter log file (in json).')
        parser.add_argument('--seed', type=int, help='Random seed for numpy')
        parser.add_argument('--args_data', type=str, help='Pickled data for stub objects')
        parser.add_argument('--snapshot_mode', type=str, default='all',
                            help='Mode to save the snapshot. Can be either "all" '
                            '(all iterations will be saved), "last" (only '
                            'the last iteration will be saved), or "none" '
                            '(do not save snapshots)')
        parser.add_argument(
            '--log_tabular_only', type=ast.literal_eval, default=False,
            help='Whether to only print the tabular log information (in a horizontal format)')

        self.update_argument_parser(parser, env_options, **kwargs)
        self.args = parser.parse_known_args(
            [arg for arg in sys.argv[2:] if arg not in ('-h', '--help')])[0]

    def rltools(self, env_options, **kwargs):
        parser = argparse.ArgumentParser()
        self.update_argument_parser(parser, self.DEFAULT_OPTS)
        self.update_argument_parser(parser, self.DEFAULT_POLICY_OPTS)

        parser.add_argument('--sampler', type=str, default='simple')
        parser.add_argument('--sampler_workers', type=int, default=1)
        parser.add_argument('--max_traj_len', type=int, default=500)
        parser.add_argument('--n_timesteps', type=int, default=12000)

        parser.add_argument('--adaptive_batch', action='store_true', default=False)
        parser.add_argument('--n_timesteps_min', type=int, default=4000)
        parser.add_argument('--n_timesteps_max', type=int, default=64000)
        parser.add_argument('--timestep_rate', type=int, default=20)

        parser.add_argument('--policy_hidden_spec', type=get_arch, default='GAE_ARCH')
        parser.add_argument('--baseline_hidden_spec', type=get_arch, default='GAE_ARCH')
        parser.add_argument('--min_std', type=float, default=1e-6)
        parser.add_argument('--max_kl', type=float, default=0.01)
        parser.add_argument('--vf_max_kl', type=float, default=0.01)
        parser.add_argument('--vf_cg_damping', type=float, default=0.01)
        parser.add_argument('--enable_obsnorm', action='store_true')
        parser.add_argument('--enable_rewnorm', action='store_true')
        parser.add_argument('--enable_vnorm', action='store_true')

        parser.add_argument('--interp_alpha', type=float, default=0.1)
        parser.add_argument('--blend_freq', type=int, default=0)
        parser.add_argument('--blend_eval_trajs', type=int, default=50)
        parser.add_argument('--keep_kmax', type=int, default=0)

        parser.add_argument('--save_freq', type=int, default=10)
        parser.add_argument('--log', type=str, required=False)
        parser.add_argument('--tblog', type=str, default='/tmp/madrl_tb_{}'.format(uuid.uuid4()))
        parser.add_argument('--no-debug', dest='debug', action='store_false')
        parser.set_defaults(debug=True)
        self.update_argument_parser(parser, env_options, **kwargs)
        self.args = parser.parse_known_args(
            [arg for arg in sys.argv[2:] if arg not in ('-h', '--help')])[0]

class Curriculum(object):

    def __init__(self, config):
        with open(config, 'r') as f:
            self.config = yaml.load(f, OrderedDictYAMLLoader)

        self._tasks = list([Task(k, v) for k, v in self.config['tasks'].items()])
        self._lesson_threshold = self.config['thresholds']['lesson']
        self._stop_threshold = self.config['thresholds']['stop']
        self._n_trials = self.config['n_trials']
        self._metric = self.config['metric']
        self._eval_trials = self.config['eval_trials']

    @property
    def tasks(self):
        return self._tasks

    @property
    def lesson_threshold(self):
        return self._lesson_threshold

    @property
    def stop_threshold(self):
        return self._stop_threshold

    @property
    def n_trials(self):
        return self._n_trials

    @property
    def metric(self):
        return self._metric

    @property
    def eval_trials(self):
        return self._eval_trials

class StandardizedEnv(AbstractMAEnv, EzPickle):

    def __init__(self, env, scale_reward=1., enable_obsnorm=False, enable_rewnorm=False,
                 obs_alpha=0.001, rew_alpha=0.001, eps=1e-8):
        EzPickle.__init__(self, env, scale_reward, enable_obsnorm, enable_rewnorm, obs_alpha,
                          rew_alpha, eps)
        self._unwrapped = env
        self._scale_reward = scale_reward
        self._enable_obsnorm = enable_obsnorm
        self._enable_rewnorm = enable_rewnorm
        self._obs_alpha = obs_alpha
        self._rew_alpha = rew_alpha
        self._eps = eps
        self._flatobs_shape = [None for _ in env.agents]
        self._obs_mean = [None for _ in env.agents]
        self._obs_var = [None for _ in env.agents]
        self._rew_mean = [None for _ in env.agents]
        self._rew_var = [None for _ in env.agents]

        for agid, agent in enumerate(env.agents):
            if isinstance(agent.observation_space, spaces.Box):
                self._flatobs_shape[agid] = np.prod(agent.observation_space.shape)
            elif isinstance(env.observation_space, spaces.Discrete):
                self._flatobs_shape[agid] = agent.observation_space.n

            self._obs_mean[agid] = np.zeros(self._flatobs_shape[agid])
            self._obs_var[agid] = np.ones(self._flatobs_shape[agid])
            self._rew_mean[agid] = 0.
            self._rew_var[agid] = 1.

    @property
    def reward_mech(self):
        return self._unwrapped.reward_mech

    @property
    def agents(self):
        return self._unwrapped.agents

    def update_obs_estimate(self, observations):
        for agid, obs in enumerate(observations):
            flatobs = np.asarray(obs).flatten()
            self._obs_mean[agid] = (1 - self._obs_alpha
                                   ) * self._obs_mean[agid] + self._obs_alpha * flatobs
            self._obs_var[agid] = (
                1 - self._obs_alpha
            ) * self._obs_var[agid] + self._obs_alpha * np.square(flatobs - self._obs_mean[agid])

    def update_rew_estimate(self, rewards):
        for agid, reward in enumerate(rewards):
            self._rew_mean[agid] = (1 - self._rew_alpha
                                   ) * self._rew_mean[agid] + self._rew_alpha * reward
            self._rew_var[agid] = (
                1 - self._rew_alpha
            ) * self._rew_var[agid] + self._rew_alpha * np.square(reward - self._rew_mean[agid])

    def standardize_obs(self, observation):
        assert isinstance(observation, list)
        self.update_obs_estimate(observation)
        return [(obs - obsmean) / (np.sqrt(obsvar) + self._eps)
                for (obs, obsmean, obsvar) in zip(observation, self._obs_mean, self._obs_var)]

    def standardize_rew(self, reward):
        assert isinstance(reward, (list, np.ndarray))
        self.update_rew_estimate(reward)
        return [
            rew / (np.sqrt(rewvar) + self._eps)
            for (rew, rewmean, rewvar) in zip(reward, self._rew_mean, self._rew_var)
        ]

    def seed(self, seed=None):
        return self._unwrapped.seed(seed)

    def reset(self):
        obs = self._unwrapped.reset()
        if self._enable_obsnorm:
            return self.standardize_obs(obs)
        else:
            return obs

    def step(self, *args):
        nobslist, rewardlist, done, info = self._unwrapped.step(*args)
        if self._enable_obsnorm:
            nobslist = self.standardize_obs(nobslist)
        if self._enable_rewnorm:
            rewardlist = self.standardize_rew(rewardlist)

        rewardlist = [self._scale_reward * rew for rew in rewardlist]
        return nobslist, rewardlist, done, info

    def __getstate__(self):
        d = EzPickle.__getstate__(self)
        d['_obs_mean'] = self._obs_mean
        d['_obs_var'] = self._obs_var
        return d

    def __setstate__(self, d):
        EzPickle.__setstate__(self, d)
        self._obs_mean = d['_obs_mean']
        self._obs_var = d['_obs_var']

    def __str__(self):
        return "Normalized {}".format(self._unwrapped)

    def render(self, *args, **kwargs):
        return self._unwrapped.render(*args, **kwargs)

    def animate(self, *args, **kwargs):
        return self._unwrapped.animate(*args, **kwargs)

class ObservationBuffer(AbstractMAEnv):

    def __init__(self, env, buffer_size):
        self._unwrapped = env
        self._buffer_size = buffer_size
        assert all([len(agent.observation_space.shape) == 1 for agent in env.agents])  # XXX
        bufshapes = [tuple(agent.observation_space.shape) + (buffer_size,) for agent in env.agents]
        self._buffer = [np.zeros(bufshape) for bufshape in bufshapes]
        self.reward_mech = self._unwrapped.reward_mech

    @property
    def agents(self):
        aglist = []
        for agid, agent in enumerate(self._unwrapped.agents):
            if isinstance(agent.observation_space, spaces.Box):
                newobservation_space = spaces.Box(low=ent.observation_space.low[0],
                                                  high=agent.observation_space.high[0],
                                                  shape=self._buffer[agid].shape)
            # elif isinstance(agent.observation_sapce, spaces.Discrete):
            else:
                raise NotImplementedError()

            aglist.append(WrappedAgent(agent, newobservation_space))

        return aglist

    @property
    def reward_mech(self):
        return self._unwrapped.reward_mech

    def seed(self, seed=None):
        return self._unwrapped.seed(seed)

    def step(self, action):
        obs, rew, done, info = self._unwrapped.step(action)
        for agid, agid_obs in enumerate(obs):
            self._buffer[agid][..., 0:self._buffer_size - 1] = self._buffer[agid][
                ..., 1:self._buffer_size].copy()
            self._buffer[agid][..., -1] = agid_obs

        bufobs = [buf.copy() for buf in self._buffer]
        return bufobs, rew, done, info

    def reset(self):
        obs = self._unwrapped.reset()

        assert isinstance(obs, list)
        for agid, agid_obs in enumerate(obs):
            for i in range(self._buffer_size):
                self._buffer[agid][..., i] = agid_obs

        bufobs = [buf.copy() for buf in self._buffer]
        return bufobs

    def render(self, *args, **kwargs):
        return self._unwrapped.render(*args, **kwargs)

    def animate(self, *args, **kwargs):
        return self._unwrapped.animate(*args, **kwargs)

class PolicyLoad(object):

    def __init__(self, env, args, max_traj_len, n_trajs, deterministic, mode='rltools'):

        self.mode = mode
        if self.mode == 'heuristic':
            self.env = env
        if self.mode == 'rltools':
            from runners.rurltools import rltools_envpolicy_parser
            self.env, self.policies, self.policy = rltools_envpolicy_parser(env, args)
        elif self.mode == 'rllab':
            from runners.rurllab import rllab_envpolicy_parser
            self.env, _ = rllab_envpolicy_parser(env, args)
            self.policy = None

        self.deterministic = deterministic
        self.max_traj_len = max_traj_len
        self.n_trajs = n_trajs
        self.disc = args['discount']
        self.control = args['control']

class Visualizer(PolicyLoad):

    def __init__(self, *args, **kwargs):
        super(Visualizer, self).__init__(*args, **kwargs)

    def __call__(self, filename, **kwargs):
        vid = kwargs.pop('vid', None)
        if self.mode == 'rltools':
            file_key = kwargs.pop('file_key', None)
            same_con_pol = kwargs.pop('same_con_pol', None)
            assert file_key
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                if self.control == 'concurrent':
                    if same_con_pol:
                        rpolicy = [self.policy] * len(self.env.agents)
                    else:
                        for pol in self.policies:
                            pol.load_h5(sess, filename, file_key)
                        rpolicy = self.policies
                        act_fns = [
                            lambda o: pol.sample_actions(o[None, ...], deterministic=self.deterministic)[0]
                            for pol in rpolicy
                        ]
                else:
                    rpolicy = self.policy.load_h5(sess, filename, file_key)
                    act_fns = [
                        lambda o: self.policy.sample_actions(o[None, ...], deterministic=self.deterministic)[0]
                    ] * len(self.env.agents)
                if vid:
                    rew, trajinfo = self.env.animate(act_fn=act_fns, nsteps=self.max_traj_len,
                                                     vid=vid, mode='rgb_array')
                else:
                    rew, trajinfo = self.env.animate(act_fn=act_fns, nsteps=self.max_traj_len)
                info = {key: np.sum(value) for key, value in trajinfo.items()}
                return (rew, info)

        if self.mode == 'rllab':
            import joblib
            from rllab.sampler.ma_sampler import cent_rollout, dec_rollout

            # XXX
            tf.reset_default_graph()
            with tf.Session() as sess:

                data = joblib.load(filename)
                policy = data['policy']
                if self.control == 'centralized':
                    path = cent_rollout(self.env, policy, max_path_length=self.max_traj_len,
                                        animated=True)
                    rew = path['rewards'].mean()
                    info = path['env_infos'].mean()
                elif self.control == 'decentralized':
                    act_fns = [lambda o: policy.get_action(o)[0]] * len(self.env.agents)
                    if vid:
                        rew, trajinfo = self.env.wrapped_env.env.animate(act_fn=act_fns,
                                                                         nsteps=self.max_traj_len,
                                                                         vid=vid, mode='rgb_array')
                    else:
                        rew, trajinfo = self.env.wrapped_env.env.animate(act_fn=act_fns,
                                                                         nsteps=self.max_traj_len)
                    info = {key: np.sum(value) for key, value in trajinfo.items()}
                return rew, info

        if self.mode == 'heuristic':
            hpolicy = kwargs.pop('hpolicy', None)
            assert hpolicy is not None
            rew, trajinfo = self.env.animate(
                act_fn=lambda o: hpolicy.sample_actions(o[None, ...])[0], nsteps=self.max_traj_len)
            info = {key: np.sum(value) for key, value in trajinfo.items()}
            return (rew, info)

# yapf: disable
ENV_OPTIONS = [
    ('radius', float, 0.015, 'Radius of agents'),
    ('n_areas_of_int', int, 8, ''),
    ('n_rovers', int, 4, ''),
    ('n_crater', int, 8, ''),
    ('n_coop', int, 2, ''),
    ('n_sensors', int, 30, ''),
    ('sensor_range', int, 0.15, ''),
    ('scout_reward', float, 5, ''),
    ('crater_reward', float, -5, ''),
    ('encounter_reward', float, 0.01, ''),
    ('reward_mech', str, 'local', ''),
    ('noid', str, None, ''),
    ('speed_features', int, 1, ''),
    ('buffer_size', int, 1, ''),
    ('curriculum', str, None, ''),
]
# yapf: enable


def main(parser):
    mode = parser._mode
    args = parser.args
    env = MAContWorld(args.n_rovers, args.n_areas_of_int, args.n_coop, args.n_crater,
                       radius=args.radius, n_sensors=args.n_sensors, scout_reward=args.scout_reward,
                       crater_reward=args.crater_reward, encounter_reward=args.encounter_reward,
                       reward_mech=args.reward_mech, sensor_range=args.sensor_range,
                       obstacle_loc=None, addid=True if not args.noid else False,
                       speed_features=bool(args.speed_features))

    if args.buffer_size > 1:
        env = ObservationBuffer(env, args.buffer_size)

    if mode == 'rllab':
        from runners.rurllab import RLLabRunner
        run = RLLabRunner(env, args)
    else:
        raise NotImplementedError()

    if args.curriculum:
        curr = Curriculum(args.curriculum)
        run(curr)
    else:
        run()


if __name__ == '__main__':
    main(RunnerParser(ENV_OPTIONS))
