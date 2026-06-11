# ATEC B2 Model Training & Submission Guide

This document outlines the end-to-end pipeline for training the Unitree B2 for the Extreme Parkour task, extracting the final PyTorch (`.pt`) model, and submitting it to the ATEC evaluation sandbox.

Because the B2 utilizes an **Asymmetric Actor-Critic (Teacher-Student)** training pipeline, you must train two separate models.

---

## Step 1: Train the Teacher Policy

The Teacher policy trains using privileged simulation information (like a perfect terrain height grid) to easily learn complex parkour skills.

Run the following command to train the Teacher:
```bash
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Teacher-B2-v0 --headless --num_envs 4096
```

> [!TIP]
> Let the Teacher train until it successfully crosses the extreme parkour terrain with high reliability. RSL-RL will automatically save checkpoints in the `logs/rsl_rl/` directory.

---

## Step 2: Train the Student Policy (Distillation)

The Student policy trains to mimic the Teacher, but it is restricted to the realistic sensors available on the real robot and the ATEC challenge (proprioception and a noisy 360-degree LiDAR).

Run the following command to train the Student:
```bash
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Student-B2-v0 --headless --num_envs 4096
```

> [!IMPORTANT]  
> The Student task (`Student-B2-v0`) will output observations in a dictionary format containing exactly two keys: `proprio` (48 dims) and `extero` (5760 dims), which perfectly aligns with what ATEC expects.

---

## Step 3: Extract the `.pt` Checkpoint

Once student training is complete or reaches a satisfactory level, you need to extract the raw PyTorch model checkpoint.

1. Navigate to your training logs directory:
   `Isaaclab_Parkour/logs/rsl_rl/Isaac-Extreme-Parkour-Student-B2-v0/`
2. Inside the folder for your latest run, look for the `.pt` files. RSL-RL saves them sequentially (e.g., `model_10000.pt`, `model_50000.pt`).
3. Copy the highest iteration `.pt` file (or `model_latest.pt`) into the ATEC workspace.
   ```bash
   cp logs/rsl_rl/Isaac-Extreme-Parkour-Student-B2-v0/<YOUR_RUN>/model_50000.pt ATEC2026_Simulation_Challenge/atec_robot_model/my_b2_student_policy.pt
   ```

---

## Step 4: Configure the ATEC Submission (`demo/solution.py`)

The ATEC evaluation script expects you to load your PyTorch model and execute it within the `demo/solution.py` script. 

Update `ATEC2026_Simulation_Challenge/demo/solution.py` with the following implementation:

```python
import torch

class AlgSolution:
    def __init__(self):
        # 1. Load your trained student policy checkpoint
        self.policy = torch.jit.load('./atec_robot_model/my_b2_student_policy.pt')
        
        # NOTE: If you are NOT exporting the model to JIT and are loading the raw state_dict, 
        # you will need to instantiate your RSL-RL Actor network architecture here, and load the 
        # state_dict manually:
        # self.policy = MyActorNetwork()
        # self.policy.load_state_dict(torch.load('...', map_location='cpu')['model_state_dict'])
        
    def predicts(self, obs, current_score):
        # 2. Extract the identical ATEC observation keys
        proprio = obs['proprio']
        extero = obs['extero']
        
        # 3. Concatenate them exactly how the IsaacLab environment flattened them during training
        # (48 dims + 5760 dims = 5808 dims)
        obs_tensor = torch.cat([proprio, extero], dim=-1)
        
        # 4. Get the continuous action vector from the policy
        action = self.policy(obs_tensor)
        
        # 5. Return the expected output dict format
        return {'action': action.tolist(), 'giveup': False}
```

---

## Step 5: Test Locally

With the model loaded and the observation inputs properly concatenated, you can test your policy locally in the ATEC challenge sandbox:

```bash
cd ATEC2026_Simulation_Challenge
python scripts/play_atec_task.py --task ATEC-TaskD-B2 --enable_cameras
```
