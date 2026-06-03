with open("scripts/rsl_rl/modules/ppo_with_extractor.py", "r") as f:
    content = f.read()

import re

old_block = """        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()"""

new_block = """        print(f"DEBUG obs type: {type(obs)}, critic_obs type: {type(critic_obs)}")
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()"""

content = content.replace(old_block, new_block)

with open("scripts/rsl_rl/modules/ppo_with_extractor.py", "w") as f:
    f.write(content)

