from __future__ import print_function
from __future__ import absolute_import

import gym
import gym.envs
import gym.spaces
from gym import spaces
import uuid
import os
import json
import pprint
import h5py
import numpy as np
import tensorflow as tf

from rllab.rllab.envs.base import Env, Step
from rllab.rllab.core.serializable import Serializable
from rllab.rllab.spaces.box import Box
from rllab.rllab.spaces.discrete import Discrete

import rltools.rltools.util


class FileHandler(object):

    def __init__(self, filename):
        self.filename = filename
        # Handle remote files
        if ':' in filename:
            datafilename = str(uuid.uuid4())
            if 'h5' in filename.split('.')[-1]:
                os.system('rsync -avrz {}.h5 /data/{}.h5'.format(
                    filename.split('.')[0], datafilename))
                newfilename = '/data/{}.{}'.format(datafilename, filename.split('.')[-1])
                self.filename = newfilename
            else:
                os.system('rsync -avrz {} /data/{}.pkl'.format(filename, datafilename))
                os.system('rsync -avrz {} /data/params.json'.format(
                    os.path.join(os.path.dirname(filename), 'params.json')))
                newfilename = '/data/{}.pkl'.format(datafilename)
                self.filename = newfilename
        # Loading file
        if 'h5' in self.filename.split('.')[-1]:
            print(self.filename.split('.'))
            self.mode = 'rltools'
            self.filename, self.file_key = rltools.util.split_h5_name(self.filename)
            print('Loading parameters from {} in {}'.format(self.file_key, filename))
            with h5py.File(self.filename, 'r') as f:
                self.train_args = json.loads(f.attrs['args'])
                dset = f[self.file_key]

        else:
            self.mode = 'rllab'
            policy_dir = os.path.dirname(self.filename)
            params_file = os.path.join(policy_dir, 'params.json')
            self.filename = self.filename
            self.file_key = None
            print('Loading parameters from {} in {}'.format('params.json', policy_dir))
            with open(params_file, 'r') as df:
                self.train_args = json.load(df)


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


class Evaluator(PolicyLoad):

    def __init__(self, *args, **kwargs):
        super(Evaluator, self).__init__(*args, **kwargs)

    def __call__(self, filename, **kwargs):
        if self.mode == 'rltools':
            file_key = kwargs.pop('file_key', None)
            same_con_pol = kwargs.pop('same_con_pol', None)
            assert file_key
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                if self.control == 'concurrent':
                    if same_con_pol:
                        rpolicy = [self.policy] * len(self.env.agents)
                    else:
                        for pol in self.policies:
                            pol.load_h5(sess, filename, file_key)
                        rpolicy = self.policies
                else:
                    self.policy.load_h5(sess, filename, file_key)
                    rpolicy = self.policy
                return rltools.util.evaluate_policy(self.env, rpolicy,
                                                    deterministic=self.deterministic,
                                                    disc=self.disc, mode=self.control,
                                                    max_traj_len=self.max_traj_len,
                                                    n_trajs=self.n_trajs)
        elif self.mode == 'rllab':
            import joblib
            import rllab.rllab.misc.evaluate
            # XXX
            tf.reset_default_graph()
            with tf.Session() as sess:
                data = joblib.load(filename)
                policy = data['policy']
                return rllab.misc.evaluate.evaluate(self.env, policy, disc=self.disc,
                                                       ma_mode=self.control,
                                                       max_path_length=self.max_traj_len,
                                                       n_paths=self.n_trajs)
        elif self.mode == 'heuristic':
            hpolicy = kwargs.pop('hpolicy', None)
            assert hpolicy
            return rltools.util.evaluate_policy(self.env, hpolicy, deterministic=self.deterministic,
                                                disc=self.disc, mode=self.control,
                                                max_traj_len=self.max_traj_len,
                                                n_trajs=self.n_trajs)


class Visualizer(PolicyLoad):

    def __init__(self, *args, **kwargs):
        super(Visualizer, self).__init__(*args, **kwargs)

    def __call__(self, filename, **kwargs):
        vid = kwargs.pop('vid', None)

        if self.mode == 'rllab':
            import joblib
            from rllab.rllab.sampler.ma_sampler import cent_rollout, dec_rollout

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

			
def convert_gym_space(space, n_agents=1):
    if isinstance(space, gym.spaces.Box) or isinstance(space, Box):
        if len(space.shape) > 1:
            assert n_agents == 1, "multi-dimensional inputs for centralized agents not supported"
            return Box(low=np.min(space.low), high=np.max(space.high), shape=space.shape)
        else:
            return Box(low=np.min(space.low), high=np.max(space.high),
                       shape=(space.shape[0] * n_agents,))
    elif isinstance(space, gym.spaces.Discrete) or isinstance(space, Discrete):
        return Discrete(n=space.n**n_agents)
    else:
        raise NotImplementedError


class RLLabEnv(Env, Serializable):

    def __init__(self, env, ma_mode):
        Serializable.quick_init(self, locals())

        self.env = env
        if hasattr(env, 'id'):
            self.env_id = env.id
        else:
            self.env_id = 'MA-Wrapper-v0'

        if ma_mode == 'centralized':
            obsfeat_space = convert_gym_space(env.agents[0].observation_space,
                                              n_agents=len(env.agents))
            action_space = convert_gym_space(env.agents[0].action_space, n_agents=len(env.agents))
        elif ma_mode in ['decentralized', 'concurrent']:
            obsfeat_space = convert_gym_space(env.agents[0].observation_space, n_agents=1)
            action_space = convert_gym_space(env.agents[0].action_space, n_agents=1)

        else:
            raise NotImplementedError

        self._observation_space = obsfeat_space
        self._action_space = action_space
        if hasattr(env, 'timestep_limit'):
            self._horizon = env.timestep_limit
        else:
            self._horizon = 250

    @property
    def agents(self):
        return self.env.agents

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        if info is None:
            info = dict()
        return Step(next_obs, reward, done, **info)

    def render(self):
        self.env.render()

    def set_param_values(self, *args, **kwargs):
        self.env.set_param_values(*args, **kwargs)

    def get_param_values(self, *args, **kwargs):
        self.env.get_param_values(*args, **kwargs)