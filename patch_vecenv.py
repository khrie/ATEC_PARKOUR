with open("scripts/rsl_rl/vecenv_wrapper.py", "r") as f:
    content = f.read()

import re
old_block = """        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim["proprio"][0]
            if "extero" in self.unwrapped.observation_manager.group_obs_dim:
                ext_dim = self.unwrapped.observation_manager.group_obs_dim["extero"]
                if len(ext_dim) > 0:
                    self.num_obs += ext_dim[0]"""

new_block = """        if hasattr(self.unwrapped, "observation_manager"):
            if "policy" in self.unwrapped.observation_manager.group_obs_dim:
                self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
            elif "proprio" in self.unwrapped.observation_manager.group_obs_dim:
                self.num_obs = self.unwrapped.observation_manager.group_obs_dim["proprio"][0]
                if "extero" in self.unwrapped.observation_manager.group_obs_dim:
                    ext_dim = self.unwrapped.observation_manager.group_obs_dim["extero"]
                    if len(ext_dim) > 0:
                        self.num_obs += ext_dim[0]
            else:
                raise KeyError("Neither 'policy' nor 'proprio' found in observation groups")"""

content = content.replace(old_block, new_block)

with open("scripts/rsl_rl/vecenv_wrapper.py", "w") as f:
    f.write(content)

