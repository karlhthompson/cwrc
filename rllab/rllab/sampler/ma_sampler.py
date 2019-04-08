import pickle
import time

import numpy as np

from rllab.rllab.misc import logger, tensor_utils
from rllab.rllab.sampler.parallel_sampler import (_get_scoped_G, _worker_set_env_params)
from rllab.rllab.sampler.stateful_pool import singleton_pool


def cent_rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1):
    """Centralized rollout"""
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        if isinstance(r, (list, np.ndarray)):
            assert (r == r[0]).all()
            r = r[0]
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated:
        env.render()
    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),)


def dec_rollout(env, agents, max_path_length=np.inf, animated=False, speedup=1):
    """Decentralized rollout"""
    n_agents = len(env.agents)
    observations = [[] for _ in range(n_agents)]
    actions = [[] for _ in range(n_agents)]
    rewards = [[] for _ in range(n_agents)]
    agent_infos = [[] for _ in range(n_agents)]
    env_infos = [[] for _ in range(n_agents)]
    olist = env.reset()
    assert len(olist) == n_agents, "{} != {}".format(len(olist), n_agents)
    agents.reset(dones=[True for _ in range(n_agents)])
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        alist, agent_info_list = agents.get_actions(olist)
        agent_info_list = tensor_utils.split_tensor_dict_list(agent_info_list)
        # For each agent
        for i, o in enumerate(olist):
            observations[i].append(env.observation_space.flatten(o))
            actions[i].append(env.action_space.flatten(alist[i]))
            if agent_info_list is None:
                agent_infos[i].append({})
            else:
                agent_infos[i].append(agent_info_list[i])

        next_olist, rlist, d, env_info = env.step(np.asarray(alist))
        for i, r in enumerate(rlist):
            rewards[i].append(r)
            env_infos[i].append(env_info)
        path_length += 1
        if d:
            break
        olist = next_olist
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated:
        env.render()

    return [
        dict(
            observations=tensor_utils.stack_tensor_list(observations[i]),
            actions=tensor_utils.stack_tensor_list(actions[i]),
            rewards=tensor_utils.stack_tensor_list(rewards[i]),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos[i]),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos[i]),) for i in range(n_agents)
    ]


def conc_rollout(env, agents, max_path_length=np.inf, animated=False, speedup=1):
    """Concurrent rollout"""
    n_agents = len(env.agents)
    observations = [[] for _ in range(n_agents)]
    actions = [[] for _ in range(n_agents)]
    rewards = [[] for _ in range(n_agents)]
    agent_infos = [[] for _ in range(n_agents)]
    env_infos = [[] for _ in range(n_agents)]
    olist = env.reset()
    for agent in agents:
        agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        alist = []
        # For each agent
        for i, o in enumerate(olist):
            a, ainfo = agents[i].get_action(o)
            alist.append(a)
            observations[i].append(env.observation_space.flatten(o))
            actions[i].append(env.action_space.flatten(a))
            if ainfo is None:
                agent_infos[i].append({})
            else:
                agent_infos[i].append(ainfo)
        next_olist, rlist, d, env_info = env.step(np.asarray(alist))
        for i, r in enumerate(rlist):
            rewards[i].append(r)
            env_infos[i].append(env_info)
        path_length += 1
        if d:
            break
        olist = next_olist
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated:
        env.render()

    return [
        dict(
            observations=tensor_utils.stack_tensor_list(observations[i]),
            actions=tensor_utils.stack_tensor_list(actions[i]),
            rewards=tensor_utils.stack_tensor_list(rewards[i]),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos[i]),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos[i]),) for i in range(n_agents)
    ]


def _worker_populate_task(G, env, policy, ma_mode, scope=None):
    # TODO: better term for both policy/policies
    G = _get_scoped_G(G, scope)
    G.env = pickle.loads(env)
    if ma_mode == 'concurrent':
        G.policies = pickle.loads(policy)
        assert isinstance(G.policies, list)
    else:
        G.policy = pickle.loads(policy)


def _worker_terminate_task(G, scope=None):
    G = _get_scoped_G(G, scope)
    if getattr(G, "env", None):
        G.env.terminate()
        G.env = None
    if getattr(G, "policy", None):
        G.policy.terminate()
        G.policy = None
    if getattr(G, "policies", None):
        for policy in G.policies:
            policy.terminate()
        G.policies = None
    if getattr(G, "sess", None):
        G.sess.close()
        G.sess = None


def populate_task(env, policy, ma_mode, scope=None):
    logger.log("Populating workers...")
    logger.log("ma_mode={}".format(ma_mode))
    if singleton_pool.n_parallel > 1:
        singleton_pool.run_each(_worker_populate_task,
                                [(pickle.dumps(env), pickle.dumps(policy), ma_mode, scope)] *
                                singleton_pool.n_parallel)
    else:
        # avoid unnecessary copying
        G = _get_scoped_G(singleton_pool.G, scope)
        G.env = env
        if ma_mode == 'concurrent':
            G.policies = policy
        else:
            G.policy = policy
    logger.log("Populated")


def terminate_task(scope=None):
    singleton_pool.run_each(_worker_terminate_task, [(scope,)] * singleton_pool.n_parallel)


def _worker_set_policy_params(G, params, ma_mode, scope=None):
    G = _get_scoped_G(G, scope)
    if ma_mode == 'concurrent':
        for pid, policy in enumerate(G.policies):
            policy.set_param_values(params[pid])
    else:
        G.policy.set_param_values(params)


def _worker_collect_path_one_env(G, max_path_length, ma_mode, scope=None):
    G = _get_scoped_G(G, scope)
    if ma_mode == 'centralized':
        path = cent_rollout(G.env, G.policy, max_path_length)
        return path, len(path['rewards'])
    elif ma_mode == 'decentralized':
        paths = dec_rollout(G.env, G.policy, max_path_length)
        lengths = [len(path['rewards']) for path in paths]
        return paths, sum(lengths)
    elif ma_mode == 'concurrent':
        paths = conc_rollout(G.env, G.policies, max_path_length)
        lengths = [len(path['rewards']) for path in paths]
        return paths, lengths[0]
    else:
        raise NotImplementedError("incorrect rollout type")


def sample_paths(policy_params, max_samples, ma_mode, max_path_length=np.inf, env_params=None,
                 scope=None):
    if ma_mode == 'concurrent':
        assert isinstance(policy_params, list)
    singleton_pool.run_each(_worker_set_policy_params,
                            [(policy_params, ma_mode, scope)] * singleton_pool.n_parallel)
    if env_params is not None:
        singleton_pool.run_each(_worker_set_env_params,
                                [(env_params, scope)] * singleton_pool.n_parallel)

    return singleton_pool.run_collect(_worker_collect_path_one_env, threshold=max_samples,
                                      args=(max_path_length, ma_mode, scope), show_prog_bar=True)
