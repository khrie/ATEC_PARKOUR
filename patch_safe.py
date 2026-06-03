with open("scripts/rsl_rl/modules/ppo_with_extractor.py", "r") as f:
    content = f.read()

import re

old_block = """        privileged_obs_type = kwargs.get("privileged_obs_type")
        if privileged_obs_type is None:
            privileged_obs_type = "critic"
        critic_obs = obs_dict.get(privileged_obs_type, obs) if isinstance(obs_dict, dict) or hasattr(obs_dict, 'keys') else obs_dict"""

new_block = """        privileged_obs_type = kwargs.get("privileged_obs_type", "critic")
        if (isinstance(obs_dict, dict) or hasattr(obs_dict, 'keys')) and privileged_obs_type is not None:
            if privileged_obs_type in obs_dict.keys():
                critic_obs = obs_dict[privileged_obs_type]
            else:
                critic_obs = obs
        else:
            critic_obs = obs if not (isinstance(obs_dict, dict) or hasattr(obs_dict, 'keys')) else obs_dict"""

content = content.replace(old_block, new_block)

with open("scripts/rsl_rl/modules/ppo_with_extractor.py", "w") as f:
    f.write(content)

