with open("scripts/rsl_rl/modules/ppo_with_extractor.py", "r") as f:
    content = f.read()

content = content.replace(
    'privileged_obs_type = kwargs.get("privileged_obs_type", "critic")',
    'privileged_obs_type = kwargs.get("privileged_obs_type")\n        if privileged_obs_type is None:\n            privileged_obs_type = "critic"'
)

with open("scripts/rsl_rl/modules/ppo_with_extractor.py", "w") as f:
    f.write(content)

