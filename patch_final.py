with open("scripts/rsl_rl/modules/ppo_with_extractor.py", "r") as f:
    content = f.read()

import re
old_block = """        print(f"DEBUG obs type: {type(obs)}, critic_obs type: {type(critic_obs)}")
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        if self.train_with_estimated_states:
            obs_est = obs.clone()
            priv_states_estimated = self.estimator(obs_est[:, :self.num_prop])
            obs_est[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.priv_states_dim] = priv_states_estimated
            self.transition.actions = self.policy.act(obs_est, **kwargs).detach()
        else:
            self.transition.actions = self.policy.act(obs, **kwargs).detach()
        
        privileged_obs_type = kwargs.get("privileged_obs_type", "critic")
        if (isinstance(obs_dict, dict) or hasattr(obs_dict, 'keys')) and privileged_obs_type is not None:
            if privileged_obs_type in obs_dict.keys():
                critic_obs = obs_dict[privileged_obs_type]
            else:
                critic_obs = obs
        else:
            critic_obs = obs if not (isinstance(obs_dict, dict) or hasattr(obs_dict, 'keys')) else obs_dict"""

new_block = """        privileged_obs_type = kwargs.get("privileged_obs_type", "critic")
        if (isinstance(obs_dict, dict) or hasattr(obs_dict, 'keys')) and privileged_obs_type is not None:
            if privileged_obs_type in obs_dict.keys():
                critic_obs = obs_dict[privileged_obs_type]
            else:
                critic_obs = obs
        else:
            critic_obs = obs if not (isinstance(obs_dict, dict) or hasattr(obs_dict, 'keys')) else obs_dict

        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        if self.train_with_estimated_states:
            obs_est = obs.clone()
            priv_states_estimated = self.estimator(obs_est[:, :self.num_prop])
            obs_est[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.priv_states_dim] = priv_states_estimated
            self.transition.actions = self.policy.act(obs_est, **kwargs).detach()
        else:
            self.transition.actions = self.policy.act(obs, **kwargs).detach()"""

content = content.replace(old_block, new_block)

with open("scripts/rsl_rl/modules/ppo_with_extractor.py", "w") as f:
    f.write(content)

