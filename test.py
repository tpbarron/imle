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


import gym
import roboschool
import roboschool_x

env = gym.make('RoboschoolAntPlain-v0')
# env = gym.make('RoboschoolHumanoidFlagrunHarderX-v0')
while True:
    i = 0
    obs = env.reset()
    done = False
    while not done and i < 100:
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())
        i += 1
