with open("scripts/rsl_rl/modules/on_policy_runner_with_extractor.py", "r") as f:
    content = f.read()

import re

old_block = """        privileged_obs = obs_dict.get(self.privileged_obs_type, obs) if (self.privileged_obs_type is not None and (isinstance(obs_dict, dict) or hasattr(obs_dict, 'get'))) else obs"""

new_block = """        if self.privileged_obs_type is not None and (isinstance(obs_dict, dict) or hasattr(obs_dict, 'keys')):
            if self.privileged_obs_type in obs_dict.keys():
                privileged_obs = obs_dict[self.privileged_obs_type]
            else:
                privileged_obs = obs
        else:
            privileged_obs = obs"""

content = content.replace(old_block, new_block)

with open("scripts/rsl_rl/modules/on_policy_runner_with_extractor.py", "w") as f:
    f.write(content)

