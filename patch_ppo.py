with open("scripts/rsl_rl/modules/ppo_with_extractor.py", "r") as f:
    content = f.read()

helper = """
def _extract_obs(obs_dict):
    if isinstance(obs_dict, dict):
        if "policy" in obs_dict:
            return obs_dict["policy"]
        elif "proprio" in obs_dict:
            import torch
            obs_parts = [obs_dict["proprio"]]
            if "extero" in obs_dict and isinstance(obs_dict["extero"], torch.Tensor) and obs_dict["extero"].numel() > 0:
                obs_parts.append(obs_dict["extero"])
            return torch.cat(obs_parts, dim=-1)
    return obs_dict

"""

content = content.replace("class PPOWithExtractor(PPO):", helper + "class PPOWithExtractor(PPO):")
content = content.replace('obs = obs_dict["policy"] if "policy" in obs_dict.keys() else obs_dict', 'obs = _extract_obs(obs_dict)')
content = content.replace('obs_batch = batch.observations["policy"] if "policy" in batch.observations.keys() else batch.observations', 'obs_batch = _extract_obs(batch.observations)')

with open("scripts/rsl_rl/modules/ppo_with_extractor.py", "w") as f:
    f.write(content)

