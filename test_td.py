import torch
from tensordict import TensorDict

td = TensorDict({"policy": torch.ones(3)}, batch_size=[])
obs = td["policy"]
print("obs type:", type(obs))
critic_obs = td.get("critic", obs)
print("critic_obs type:", type(critic_obs))

