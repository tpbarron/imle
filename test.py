# import pybullet as p
# p.connect(p.GUI)
#
# from envs.gym_manipulators import PybulletReacher
# env = PybulletReacher()
#
# obs = env.reset()
# for i in range(10):
#     print ("Env: ", i)
#     done = False
#     while not done:
#         print ("Stepping")
#         env.render()
#         obs, rew, done, info = env.step(env.action_space.sample())


import numpy as np
import gym
# import pybullet_envs
import gym_x

# env = gym.make('MountainCarContinuousVisionX-v0')
env = gym.make('Walker2DBulletX-v0')
# env = gym.make('Walker2DVisionBulletX-v0')
# env = gym.make('ChainX-v0')
# env = gym.make('ChainVisionX-v0')
# env = gym.make('HopperBulletX-v0')
# env = gym.make('HopperVisionBulletX-v0')
# env = gym.make('HalfCheetahVisionBulletX-v0')
# env = gym.make('InvertedPendulumSwingupVisionBulletEnv-v0')
# env = gym.make('AcrobotContinuousVisionX-v0')
# env = gym.make('MountainCarContinuousVisionX-v0')

env.render(mode='human')
for i in range(10):
    i = 0
    obs = env.reset()
    done = False
    while not done and i < 100:
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())
        print (obs.min(), obs.max(), obs.dtype)

        # if i == 50:
        #     obs = obs.astype(np.uint8)
        #     print (obs.dtype)
        #     from PIL import Image
        #     dims = (64, 64) #32, 32)
        #     I = Image.fromarray(obs.reshape(dims))
        #     I.show()
        #     input("")

        # import time
        # time.sleep(0.1)
        i += 1
env.close()
