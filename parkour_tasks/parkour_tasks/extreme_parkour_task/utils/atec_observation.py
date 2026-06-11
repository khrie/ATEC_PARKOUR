from __future__ import annotations
import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
from isaaclab.sensors import RayCaster
from isaaclab.assets import Articulation
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

class AtecTeacherObservations(ManagerTermBase):
    """
    Custom observation term for the Teacher that exactly matches the ATEC proprioception
    (48 dimensions) and adds scandots (132 dimensions), with NO history.
    """
    def __init__(self, cfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.ray_sensor: RayCaster = env.scene.sensors['height_scanner']
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.measured_heights = torch.zeros(self.num_envs, 132, device=self.device)
        self.env = env

    def reset(self, env_ids=None) -> None:
        pass

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        
        # 1. Base Linear Velocity (3)
        base_lin_vel = self.asset.data.root_lin_vel_b
        
        # 2. Base Angular Velocity (3)
        base_ang_vel = self.asset.data.root_ang_vel_b
        
        # 3. Velocity Commands (3)
        commands = env.command_manager.get_command('base_velocity')
        
        # 4. Projected Gravity (3)
        projected_gravity = self.asset.data.projected_gravity_b
        
        # 5. Joint Positions (relative) (12)
        joint_pos = self.asset.data.joint_pos - self.asset.data.default_joint_pos
        
        # 6. Joint Velocities (relative) (12)
        joint_vel = self.asset.data.joint_vel - self.asset.data.default_joint_vel
        
        # 7. Last Actions (12)
        actions = env.action_manager.action

        # Concatenate proprio to perfectly match ATEC ProprioObservationsCfg (48 dims)
        proprio = torch.cat((
            base_lin_vel,
            base_ang_vel,
            commands,
            projected_gravity,
            joint_pos,
            joint_vel,
            actions
        ), dim=-1)

        # Update height scan every 5 steps
        if env.common_step_counter % 5 == 0:
            self.measured_heights = self._get_heights()

        # Concatenate proprio + scandots (180 dims total)
        observations = torch.cat([proprio, self.measured_heights], dim=-1)
        return observations 

    def _get_heights(self):
        return torch.clip(self.ray_sensor.data.pos_w[:, 2].unsqueeze(1) - self.ray_sensor.data.ray_hits_w[..., 2] - 0.3, -1, 1).to(self.device)


class AtecTeacherProprio(ManagerTermBase):
    """
    Custom observation term for the Teacher that exactly matches the ATEC proprioception
    (48 dimensions), with NO history.
    """
    def __init__(self, cfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.env = env

    def reset(self, env_ids=None) -> None:
        pass

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        
        # 1. Base Linear Velocity (3)
        base_lin_vel = self.asset.data.root_lin_vel_b
        
        # 2. Base Angular Velocity (3)
        base_ang_vel = self.asset.data.root_ang_vel_b
        
        # 3. Velocity Commands (3)
        commands = env.command_manager.get_command('base_velocity')
        
        # 4. Projected Gravity (3)
        projected_gravity = self.asset.data.projected_gravity_b
        
        # 5. Joint Positions (relative) (12)
        joint_pos = self.asset.data.joint_pos - self.asset.data.default_joint_pos
        
        # 6. Joint Velocities (relative) (12)
        joint_vel = self.asset.data.joint_vel - self.asset.data.default_joint_vel
        
        # 7. Last Actions (12)
        actions = env.action_manager.action

        # Concatenate proprio to perfectly match ATEC ProprioObservationsCfg (48 dims)
        proprio = torch.cat((
            base_lin_vel,
            base_ang_vel,
            commands,
            projected_gravity,
            joint_pos,
            joint_vel,
            actions
        ), dim=-1)

        return proprio 


class AtecTeacherExtero(ManagerTermBase):
    """
    Custom observation term for the Teacher that exactly matches the ATEC scandots
    (132 dimensions), with NO history. Updates every 5 steps.
    """
    def __init__(self, cfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.ray_sensor: RayCaster = env.scene.sensors['height_scanner']
        self.measured_heights = torch.zeros(self.num_envs, 132, device=self.device)
        self.env = env

    def reset(self, env_ids=None) -> None:
        pass

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        
        # Update height scan every 5 steps
        if env.common_step_counter % 5 == 0:
            self.measured_heights = self._get_heights()

        return self.measured_heights 

    def _get_heights(self):
        return torch.clip(self.ray_sensor.data.pos_w[:, 2].unsqueeze(1) - self.ray_sensor.data.ray_hits_w[..., 2] - 0.3, -1, 1).to(self.device)
