import numpy as np
import scipy.spatial.distance as ssd
import distutils.spawn, distutils.version
from libavwrapper import AVConv
import logging
import subprocess
import gym
from gym import spaces
from gym import error
from gym.utils import seeding
gym.logger.set_level(40)
logger = logging.getLogger(__name__)

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

class EzPickle(object):

    def __init__(self, *args, **kwargs):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs

    def __getstate__(self):
        return {"_ezpickle_args": self._ezpickle_args, "_ezpickle_kwargs": self._ezpickle_kwargs}

    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)


def stack_dict_list(dict_list):
    ret = dict()
    if not dict_list:
        return ret
    keys = dict_list[0].keys()
    for k in keys:
        eg = dict_list[0][k]
        if isinstance(eg, dict):
            v = stack_dict_list([x[k] for x in dict_list])
        else:
            v = np.array([x[k] for x in dict_list])
        ret[k] = v

    return ret


class ImageEncoder(object):
    def __init__(self, output_path, frame_shape, frames_per_sec):
        self.proc = None
        self.output_path = output_path
        # Frame shape should be lines-first, so w and h are swapped
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise error.InvalidFrame("Your frame has shape {}, but we require (w,h,3) or (w,h,4), i.e. RGB values for a w-by-h image, with an optional alpha channl.".format(frame_shape))
        self.wh = (w,h)
        self.includes_alpha = (pixfmt == 4)
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec

        if distutils.spawn.find_executable('./visualization/ffmpeg') is not None:
            self.backend = './visualization/ffmpeg'
        elif distutils.spawn.find_executable('avconv') is not None:
            self.backend = 'avconv'
        else:
            raise error.DependencyNotInstalled("""Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.""")

        self.start()

    @property
    def version_info(self):
        return {
            'backend':self.backend,
            'version':str(subprocess.check_output([self.backend, '-version'],
                                                  stderr=subprocess.STDOUT)),
            'cmdline':self.cmdline
        }

    def start(self):
        self.cmdline = (self.backend,
                     '-nostats',
                     '-loglevel', 'error', # suppress warnings
                     '-y',
                     '-r', '%d' % self.frames_per_sec,

                     # input
                     '-f', 'rawvideo',
                     '-s:v', '{}x{}'.format(*self.wh),
                     '-pix_fmt',('rgb32' if self.includes_alpha else 'rgb24'),
                     '-i', '-', # this used to be /dev/stdin, which is not Windows-friendly

                     # output
                     '-vcodec', 'libx264',
                     '-pix_fmt', 'yuv420p',
                     self.output_path
                     )

        logger.debug('Starting ffmpeg with "%s"', ' '.join(self.cmdline))
        self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise error.InvalidFrame('Wrong type {} for {} (must be np.ndarray or np.generic)'.format(type(frame), frame))
        if frame.shape != self.frame_shape:
            raise error.InvalidFrame("Your frame has shape {}, but the VideoRecorder is configured for shape {}.".format(frame.shape, self.frame_shape))
        if frame.dtype != np.uint8:
            raise error.InvalidFrame("Your frame has data type {}, but we require uint8 (i.e. RGB values from 0-255).".format(frame.dtype))

        if distutils.version.LooseVersion(np.__version__) >= distutils.version.LooseVersion('1.9.0'):
            self.proc.stdin.write(frame.tobytes())
        else:
            self.proc.stdin.write(frame.tostring())

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            logger.error("VideoRecorder encoder exited with status {}".format(ret))


class Agent(object):

    def __new__(cls, *args, **kwargs):
        agent = super(Agent, cls).__new__(cls)
        return agent

    @property
    def observation_space(self):
        raise NotImplementedError()

    @property
    def action_space(self):
        raise NotImplementedError()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)


class AbstractMAEnv(object):

    def __new__(cls, *args, **kwargs):
        # Easier to override __init__
        env = super(AbstractMAEnv, cls).__new__(cls)
        env._unwrapped = None
        return env

    def setup(self):
        pass

    def seed(self, seed=None):
        return []

    @property
    def agents(self):
        """Returns the agents in the environment. List of objects inherited from Agent class

        Should give us information about cooperating and competing agents?
        """
        raise NotImplementedError()

    @property
    def reward_mech(self):
        raise NotImplementedError()

    def reset(self):
        """Resets the game"""
        raise NotImplementedError()

    def step(self, actions):
        raise NotImplementedError()

    @property
    def is_terminal(self):
        raise NotImplementedError()

    def set_param_values(self, lut):
        for k, v in lut.items():
            setattr(self, k, v)
        self.setup()

    def render(self, *args, **kwargs):
        raise NotImplementedError()

    def animate(self, act_fn, nsteps, **kwargs):
        """act_fn could be a list of functions for each agent in the environemnt that we can control"""
        if not isinstance(act_fn, list):
            act_fn = [act_fn for _ in range(len(self.agents))]
        assert len(act_fn) == len(self.agents)
        encoder = None
        vid_loc = kwargs.pop('vid', None)
        obs = self.reset()
        frame = self.render(**kwargs)
        if vid_loc:
            fps = kwargs.pop('fps', 10)
            encoder = ImageEncoder(vid_loc, frame.shape, fps)
            try:
                encoder.capture_frame(frame)
            except error.InvalidFrame as e:
                print('Invalid video frame, {}'.format(e))

        rew = np.zeros((len(self.agents)))
        traj_info_list = []
        for step in range(nsteps):
            a = list(map(lambda afn, o: afn(o), act_fn, obs))
            obs, r, done, info = self.step(a)
            rew += r
            if info:
                traj_info_list.append(info)

            frame = self.render(**kwargs)
            if vid_loc:
                try:
                    encoder.capture_frame(frame)
                except error.InvalidFrame as e:
                    print('Invalid video frame, {}'.format(e))

            if done:
                break

        traj_info = stack_dict_list(traj_info_list)
        return rew, traj_info

    @property
    def unwrapped(self):
        if self._unwrapped is not None:
            return self._unwrapped
        else:
            return self

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)


class Archea(Agent):

    def __init__(self, idx, radius, n_sensors, sensor_range, addid=True, speed_features=True):
        self._idx = idx
        self._radius = radius
        self._n_sensors = n_sensors
        self._sensor_range = sensor_range
        # Number of observation coordinates from each sensor
        self._sensor_obscoord = 4
        if speed_features:
            self._sensor_obscoord += 3
        self._obscoord_from_sensors = self._n_sensors * self._sensor_obscoord
        self._obs_dim = self._obscoord_from_sensors + 2  #+ 1  #2 for type, 1 for id
        if addid:
            self._obs_dim += 1

        self._position = None
        self._velocity = None
        # Sensors
        angles_K = np.linspace(0., 2. * np.pi, self._n_sensors + 1)[:-1]
        sensor_vecs_K_2 = np.c_[np.cos(angles_K), np.sin(angles_K)]
        self._sensors = sensor_vecs_K_2

    @property
    def observation_space(self):
        return spaces.Box(low=-10, high=10, shape=(self._obs_dim,))

    @property
    def action_space(self):
        return spaces.Box(low=-1, high=1, shape=(2,))

    @property
    def position(self):
        assert self._position is not None
        return self._position

    @property
    def velocity(self):
        assert self._velocity is not None
        return self._velocity

    def set_position(self, x_2):
        assert x_2.shape == (2,)
        self._position = x_2

    def set_velocity(self, v_2):
        assert v_2.shape == (2,)
        self._velocity = v_2

    @property
    def sensors(self):
        assert self._sensors is not None
        return self._sensors

    def sensed(self, objx_N_2, same=False):
        """Whether `obj` would be sensed by the rovers"""
        relpos_obj_N_2 = objx_N_2 - np.expand_dims(self.position, 0)
        sensorvals_K_N = self.sensors.dot(relpos_obj_N_2.T)
        sensorvals_K_N[(sensorvals_K_N < 0) | (sensorvals_K_N > self._sensor_range) | ((
            relpos_obj_N_2**2).sum(axis=1)[None, :] - sensorvals_K_N**2 > self._radius**2)] = np.inf
        if same:
            sensorvals_K_N[:, self._idx - 1] = np.inf
        return sensorvals_K_N


class MAContWorld(AbstractMAEnv, EzPickle):

    def __init__(self, n_rovers=4, n_areas_of_int=8, n_coop=2, n_crater=10, radius=0.015,
                 obstacle_radius=0.07, obstacle_loc=None, n_sensors=30, 
                 sensor_range=0.15, action_scale=0.001, crater_reward=-5., scout_reward=5., 
                 encounter_reward=.01, control_penalty=-.1, reward_mech='local', addid=True, 
                 speed_features=True, **kwargs):
        EzPickle.__init__(self, n_rovers, n_areas_of_int, n_coop, n_crater, radius, obstacle_radius,
                          obstacle_loc, n_sensors, sensor_range,
                          action_scale, crater_reward, scout_reward, encounter_reward,
                          control_penalty, reward_mech, addid, speed_features, **kwargs)
        self.n_rovers = n_rovers
        self.n_areas_of_int = n_areas_of_int
        self.n_coop = n_coop
        self.n_crater = n_crater
        self.obstacle_radius = obstacle_radius
        self.obstacle_loc = obstacle_loc
        self.radius = radius
        self.n_sensors = n_sensors
        self.sensor_range = np.ones(self.n_rovers) * sensor_range
        self.action_scale = action_scale
        self.crater_reward = crater_reward
        self.scout_reward = scout_reward
        self.control_penalty = control_penalty
        self.encounter_reward = encounter_reward

        self.n_obstacles = 2
        self._reward_mech = reward_mech
        self._addid = addid
        self._speed_features = speed_features
        self.seed()
        self._rovers = [
            Archea(npu + 1, self.radius, self.n_sensors, self.sensor_range[npu], addid=self._addid,
                   speed_features=self._speed_features) for npu in range(self.n_rovers)
        ]
        self._areas_of_int = [
            Archea(nev + 1, self.radius * 2, self.n_rovers, self.sensor_range.mean() / 2)
            for nev in range(self.n_areas_of_int)
        ]
        self._craters = [
            Archea(npo + 1, self.radius * 3 / 2, self.n_crater, 0) for npo in range(self.n_crater)
        ]

    @property
    def reward_mech(self):
        return self._reward_mech

    @property
    def timestep_limit(self):
        return 1000

    @property
    def agents(self):
        return self._rovers

    def get_param_values(self):
        return self.__dict__

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def _respawn(self, objposition, objradius):
        while ssd.cdist(objposition[None, :], self.obstaclesx_No_2[[0]]) <= objradius * 2 + self.obstacle_radius:
            objposition = self.np_random.rand(2)
        return objposition

    def reset(self):
        self._timesteps = 0
        # Initialize obstacles
        if self.obstacle_loc is None:
            self.obstaclesx_No_2 = self.np_random.rand(self.n_obstacles, 2)
        else:
            self.obstaclesx_No_2 = self.obstacle_loc[None, :]
        self.obstaclesv_No_2 = np.zeros((self.n_obstacles, 2))

        # Initialize rovers
        for rover in self._rovers:
            rover.set_position(self.np_random.rand(2))
            # Avoid spawning where the obstacles lie
            rover.set_position(self._respawn(rover.position, rover._radius))
            rover.set_velocity(np.zeros(2))

        # Initialize areas_of_int
        for area in self._areas_of_int:
            area.set_position(self.np_random.rand(2))
            area.set_position(self._respawn(area.position, area._radius))

        # Initialize craters
        for crater in self._craters:
            crater.set_position(self.np_random.rand(2))
            crater.set_position(self._respawn(crater.position, crater._radius))

        return self.step(np.zeros((self.n_rovers, 2)))[0]

    @property
    def is_terminal(self):
        if self._timesteps >= self.timestep_limit:
            return True
        return False

    def _caught(self, is_colliding_N1_N2, n_coop):
        """
        Check whether collision results in observing the area of interest

        This is because you need `n_coop` rovers to collide with the area of interest at the same time to observe it
        """
        # number of N1 colliding with given N2
        n_collisions_N2 = is_colliding_N1_N2.sum(axis=0)
        is_caught_cN2 = np.where(n_collisions_N2 >= n_coop)[0]

        # number of N2 colliding with given N1
        who_collisions_N1_cN2 = is_colliding_N1_N2[:, is_caught_cN2]
        who_caught_cN1 = np.where(who_collisions_N1_cN2 >= 1)[0]

        return is_caught_cN2, who_caught_cN1

    def _closest_dist(self, closest_obj_idx_Np_K, sensorvals_Np_K_N):
        """Closest distances according to `idx`"""
        sensorvals = []
        for inp in range(self.n_rovers):
            sensorvals.append(sensorvals_Np_K_N[inp, ...][np.arange(self.n_sensors),
                                                          closest_obj_idx_Np_K[inp, ...]])
        return np.c_[sensorvals]

    def _extract_speed_features(self, objv_N_2, closest_obj_idx_N_K, sensedmask_obj_Np_K):
        sensorvals = []
        for rover in self._rovers:
            sensorvals.append(
                rover.sensors.dot((objv_N_2 - np.expand_dims(rover.velocity, 0)).T))
        sensed_objspeed_Np_K_N = np.c_[sensorvals]

        sensed_objspeedfeatures_Np_K = np.zeros((self.n_rovers, self.n_sensors))

        sensorvals = []
        for inp in range(self.n_rovers):
            sensorvals.append(sensed_objspeed_Np_K_N[inp, :, :][np.arange(self.n_sensors),
                                                                closest_obj_idx_N_K[inp, :]])
        sensed_objspeedfeatures_Np_K[sensedmask_obj_Np_K] = np.c_[sensorvals][sensedmask_obj_Np_K]

        return sensed_objspeedfeatures_Np_K

    def step(self, action_Np2):
        action_Np2 = np.asarray(action_Np2)
        action_Np_2 = action_Np2.reshape((self.n_rovers, 2))
        # Players
        actions_Np_2 = action_Np_2 * self.action_scale

        rewards = np.zeros((self.n_rovers,))
        assert action_Np_2.shape == (self.n_rovers, 2)

        for npu, rover in enumerate(self._rovers):
            rover.set_velocity(rover.velocity + actions_Np_2[npu])
            rover.set_position(rover.position + rover.velocity)

#        # Penalize large actions
#        if self.reward_mech == 'global':
#            rewards += self.control_penalty * (actions_Np_2**2).sum()
#        else:
#            rewards += self.control_penalty * (actions_Np_2**2).sum(axis=1)

        # Rovers stop on hitting a wall
        for npu, rover in enumerate(self._rovers):
            clippedx_2 = np.clip(rover.position, 0, 1)
            vel_2 = rover.velocity
            vel_2[rover.position != clippedx_2] = 0
            rover.set_velocity(vel_2)
            rover.set_position(clippedx_2)

        obstacle_coll_Np = np.zeros(self.n_rovers)
        # Rovers rebound on hitting an obstacle
        for npu, rover in enumerate(self._rovers):
            distfromobst_No = ssd.cdist(np.expand_dims(rover.position, 0), self.obstaclesx_No_2)
            is_colliding_No = distfromobst_No <= rover._radius + self.obstacle_radius
            obstacle_coll_Np[npu] = is_colliding_No.sum()
            if obstacle_coll_Np[npu] > 0:
                rover.set_velocity(-1 / 2 * rover.velocity)


        # Find collisions
        roversx_Np_2 = np.array([rover.position for rover in self._rovers])
        areas_of_intx_Ne_2 = np.array([area.position for area in self._areas_of_int])
        craterx_Npo_2 = np.array([crater.position for crater in self._craters])

        # areas_of_int
        evdists_Np_Ne = ssd.cdist(roversx_Np_2, areas_of_intx_Ne_2)
        is_colliding_area_Np_Ne = evdists_Np_Ne <= np.asarray([
            rover._radius + area._radius for rover in self._rovers
            for area in self._areas_of_int
        ]).reshape(self.n_rovers, self.n_areas_of_int)

        # num_collisions depends on how many rovers are needed to scout an area
        area_scouted, which_rover_scouted_area = self._caught(is_colliding_area_Np_Ne, self.n_coop)

        # craters
        podists_Np_Npo = ssd.cdist(roversx_Np_2, craterx_Npo_2)
        is_colliding_crat_Np_Npo = podists_Np_Npo <= np.asarray([
            rover._radius + crater._radius for rover in self._rovers
            for crater in self._craters
        ]).reshape(self.n_rovers, self.n_crater)
        crat_caught, which_rover_caught_crater = self._caught(is_colliding_crat_Np_Npo, 1)

        # Find sensed objects
        # Obstacles
        sensorvals_Np_K_No = np.array(
            [rover.sensed(self.obstaclesx_No_2) for rover in self._rovers])

        # Areas of interest
        sensorvals_Np_K_Ne = np.array([rover.sensed(areas_of_intx_Ne_2) for rover in self._rovers])

        # crater
        sensorvals_Np_K_Npo = np.array(
            [rover.sensed(craterx_Npo_2) for rover in self._rovers])

        # Allies
        sensorvals_Np_K_Np = np.array(
            [rover.sensed(roversx_Np_2, same=True) for rover in self._rovers])

        # dist features
        closest_ob_idx_Np_K = np.argmin(sensorvals_Np_K_No, axis=2)
        closest_ob_dist_Np_K = self._closest_dist(closest_ob_idx_Np_K, sensorvals_Np_K_No)
        sensedmask_ob_Np_K = np.isfinite(closest_ob_dist_Np_K)
        sensed_obdistfeatures_Np_K = np.zeros((self.n_rovers, self.n_sensors))
        sensed_obdistfeatures_Np_K[sensedmask_ob_Np_K] = closest_ob_dist_Np_K[sensedmask_ob_Np_K]
        # Areas of interest
        closest_area_idx_Np_K = np.argmin(sensorvals_Np_K_Ne, axis=2)
        closest_area_dist_Np_K = self._closest_dist(closest_area_idx_Np_K, sensorvals_Np_K_Ne)
        sensedmask_area_Np_K = np.isfinite(closest_area_dist_Np_K)
        sensed_areadistfeatures_Np_K = np.zeros((self.n_rovers, self.n_sensors))
        sensed_areadistfeatures_Np_K[sensedmask_area_Np_K] = closest_area_dist_Np_K[sensedmask_area_Np_K]
        # crater
        closest_crat_idx_Np_K = np.argmin(sensorvals_Np_K_Npo, axis=2)
        closest_crat_dist_Np_K = self._closest_dist(closest_crat_idx_Np_K, sensorvals_Np_K_Npo)
        sensedmask_crat_Np_K = np.isfinite(closest_crat_dist_Np_K)
        sensed_cratdistfeatures_Np_K = np.zeros((self.n_rovers, self.n_sensors))
        sensed_cratdistfeatures_Np_K[sensedmask_crat_Np_K] = closest_crat_dist_Np_K[sensedmask_crat_Np_K]
        # Allies
        closest_ally_idx_Np_K = np.argmin(sensorvals_Np_K_Np, axis=2)
        closest_ally_dist_Np_K = self._closest_dist(closest_ally_idx_Np_K, sensorvals_Np_K_Np)
        sensedmask_ally_Np_K = np.isfinite(closest_ally_dist_Np_K)
        sensed_allydistfeatures_Np_K = np.zeros((self.n_rovers, self.n_sensors))
        sensed_allydistfeatures_Np_K[sensedmask_ally_Np_K] = closest_ally_dist_Np_K[sensedmask_ally_Np_K]

        # speed features
        roversv_Np_2 = np.array([rover.velocity for rover in self._rovers])
        areasv_Ne_2 = np.array([rover.velocity*0 for area in self._areas_of_int]) # Pseudo speed feature for areas of interest
        craterv_Npo_2 = np.array([rover.velocity*0 for crater in self._craters]) # Pseudo speed feature for craters

        # Allies
        sensed_allyspeedfeatures_Np_K = self._extract_speed_features(roversv_Np_2,
                                                                   closest_ally_idx_Np_K,
                                                                   sensedmask_ally_Np_K)
                                                                   
        # Areas of interest
        sensed_areaspeedfeatures_Np_K = self._extract_speed_features(areasv_Ne_2,
                                                                   closest_area_idx_Np_K,
                                                                   sensedmask_area_Np_K)
        # Craters
        sensed_cratspeedfeatures_Np_K = self._extract_speed_features(craterv_Npo_2,
                                                                   closest_crat_idx_Np_K,
                                                                   sensedmask_crat_Np_K)

        # Process collisions
        # If object collided with required number of players, reset its position and velocity
        # Effectively the same as removing it and adding it back
        if area_scouted.size:
            for arscout in area_scouted:
                self._areas_of_int[arscout].set_position(self.np_random.rand(2))
                self._areas_of_int[arscout].set_position(
                    self._respawn(self._areas_of_int[arscout].position, self._areas_of_int[arscout]
                                  ._radius))

        if crat_caught.size:
            for cratcaught in crat_caught:
                self._craters[cratcaught].set_position(self.np_random.rand(2))
                self._craters[cratcaught].set_position(
                    self._respawn(self._craters[cratcaught].position, self._craters[cratcaught]
                                  ._radius))

        area_encounters, which_rover_encounterd_area = self._caught(is_colliding_area_Np_Ne, 1)

        # Update reward based on these collisions
        if self.reward_mech == 'global':
            rewards += (
                (len(area_scouted) * self.scout_reward) + (len(crat_caught) * self.crater_reward) +
                (len(area_encounters) * self.encounter_reward))
        else:
            rewards[which_rover_scouted_area] += self.scout_reward
            rewards[which_rover_caught_crater] += self.crater_reward
            rewards[which_rover_encounterd_area] += self.encounter_reward

        # Add features together
        if self._speed_features:
            sensorfeatures_Np_K_O = np.c_[sensed_obdistfeatures_Np_K, sensed_areadistfeatures_Np_K,
                                          sensed_areaspeedfeatures_Np_K, sensed_cratdistfeatures_Np_K,
                                          sensed_cratspeedfeatures_Np_K, sensed_allydistfeatures_Np_K,
                                          sensed_allyspeedfeatures_Np_K]
        else:
            sensorfeatures_Np_K_O = np.c_[sensed_obdistfeatures_Np_K, sensed_areadistfeatures_Np_K,
                                          sensed_cratdistfeatures_Np_K, sensed_allydistfeatures_Np_K]

        obslist = []
        for inp in range(self.n_rovers):
            if self._addid:
                obslist.append(
                    np.concatenate([
                        sensorfeatures_Np_K_O[inp, ...].ravel(), [
                            float((is_colliding_area_Np_Ne[inp, :]).sum() > 0), float((
                                is_colliding_crat_Np_Npo[inp, :]).sum() > 0)
                        ], [inp + 1]
                    ]))
            else:
                obslist.append(
                    np.concatenate([
                        sensorfeatures_Np_K_O[inp, ...].ravel(), [
                            float((is_colliding_area_Np_Ne[inp, :]).sum() > 0), float((
                                is_colliding_crat_Np_Npo[inp, :]).sum() > 0)
                        ]
                    ]))

        assert all([
            obs.shape == agent.observation_space.shape for obs, agent in zip(obslist, self.agents)
        ])
        self._timesteps += 1
        done = self.is_terminal
        info = dict(areascouts=len(area_scouted), cratercatches=len(crat_caught))
        return obslist, rewards, done, info

    def render(self, screen_size=700, rate=1, mode='human'):
        import cv2
        img = np.empty((screen_size, screen_size, 3), dtype=np.uint8)
        img[...] = 255
        # Obstacles
        for iobs, obstaclex_2 in enumerate(self.obstaclesx_No_2):
            assert obstaclex_2.shape == (2,)
            color = (128, 128, 0)
            cv2.circle(img,
                       tuple((obstaclex_2 * screen_size).astype(int)),
                       int(self.obstacle_radius * screen_size), color, -1, lineType=cv2.LINE_AA)
        # Rovers
        for rover in self._rovers:
            for k in range(rover._n_sensors):
                color = (0, 0, 0)
                cv2.line(img,
                         tuple((rover.position * screen_size).astype(int)),
                         tuple(((rover.position + rover._sensor_range * rover.sensors[k]) *
                                screen_size).astype(int)), color, 1, lineType=cv2.LINE_AA)
                # cv2.circle(img,
                #            tuple((rover.position * screen_size).astype(int)),
                #            int(rover._radius * screen_size), (255, 0, 0), 1, lineType=cv2.LINE_AA)
                cv2.rectangle(img,
                           tuple((rover.position * screen_size + rover._radius * screen_size).astype(int)),
                           tuple((rover.position * screen_size - rover._radius * screen_size).astype(int)), 
                           (255, 0, 0), -1)
                cv2.circle(img,
                           tuple((rover.position * screen_size).astype(int)),
                           int(rover._radius * screen_size * rover._sensor_range * 67.5), 
                                (255, 0, 0), 1, lineType=cv2.LINE_AA)
        # Areas of interest
        for area in self._areas_of_int:
            color = (0, 255, 0)
            cv2.circle(img,
                       tuple((area.position * screen_size).astype(int)),
                       int(area._radius * screen_size), color, -1, lineType=cv2.LINE_AA)

        # crater
        for crater in self._craters:
            color = (0, 0, 255)
            cv2.circle(img,
                       tuple((crater.position * screen_size).astype(int)),
                       int(crater._radius * screen_size), color, -1, lineType=cv2.LINE_AA)

        opacity = 0.4
        bg = np.ones((screen_size, screen_size, 3), dtype=np.uint8) * 255
        cv2.addWeighted(bg, opacity, img, 1 - opacity, 0, img)
        cv2.imshow('ContinuousWorld', img)
        cv2.waitKey(rate)
        return np.asarray(img)[..., ::-1]


if __name__ == '__main__':
    env = MAContWorld(5, 10, obstacle_loc=None)
    obs = env.reset()
    while True:
        obs, rew, _, _ = env.step(env.np_random.randn(10) * .5)
        if rew.sum() > 0:
            print(rew)
        env.render()
