import torch
import numpy as np
import gym
model = torch.jit.load('model_scripted.pt')
print(model.eval())

env = gym.make("CartPole-v1")
env.reset()

for i in range(20000):
    state_ = np.array(env.env.state)
    state = torch.from_numpy(state_).float()
    logits,value = model(state)
    action_dist = torch.distributions.Categorical(logits=logits)
    action = action_dist.sample()
    state2, reward, done, info = env.step(action.detach().numpy())
    if done:
        print("Lost")
        env.reset()
    state_ = np.array(env.env.state)
    state = torch.from_numpy(state_).float()
    env.render()
