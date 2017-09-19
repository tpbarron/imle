import pybullet as p
p.connect(p.GUI)

from envs.gym_manipulators import PybulletReacher
env = PybulletReacher()

obs = env.reset()
for i in range(10):
    print ("Env: ", i)
    done = False
    while not done:
        print ("Stepping")
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())
