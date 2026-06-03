with open("scripts/rsl_rl/modules/on_policy_runner_with_extractor.py", "r") as f:
    content = f.read()

content = content.replace("if isinstance(obs_dict, dict):", "if isinstance(obs_dict, dict) or hasattr(obs_dict, 'keys'):")

with open("scripts/rsl_rl/modules/on_policy_runner_with_extractor.py", "w") as f:
    f.write(content)

with open("scripts/rsl_rl/modules/ppo_with_extractor.py", "r") as f:
    content = f.read()

content = content.replace("if isinstance(obs_dict, dict):", "if isinstance(obs_dict, dict) or hasattr(obs_dict, 'keys'):")

with open("scripts/rsl_rl/modules/ppo_with_extractor.py", "w") as f:
    f.write(content)

