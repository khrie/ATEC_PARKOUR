from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.envs import mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs.mdp.events import ( 
randomize_rigid_body_mass,
apply_external_force_torque,
reset_joints_by_scale

)
from isaaclab.envs.mdp.rewards import undesired_contacts
from parkour_isaaclab.envs.mdp.parkour_actions import DelayedJointPositionActionCfg 
from parkour_tasks.extreme_parkour_task.utils.atec_observation import AtecTeacherObservations
from parkour_isaaclab.envs.mdp import terminations, rewards, parkours, events, observations, parkour_commands

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = parkour_commands.ParkourCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0,6.0 ),
        heading_control_stiffness=0.8,
        ranges=parkour_commands.ParkourCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.8), 
            heading=(-1.6, 1.6)
        ),
        clips= parkour_commands.ParkourCommandCfg.Clips(
            lin_vel_clip = 0.2,
            ang_vel_clip = 0.4
        )
    )

@configclass
class ParkourEventsCfg:
    """Command specifications for the MDP."""
    base_parkour = parkours.ParkourEventsCfg(
        asset_name = 'robot',
        )

@configclass
class TeacherObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        teacher = ObsTerm(
            func=AtecTeacherObservations,
            params={
                "asset_cfg":SceneEntityCfg("robot"),
            }
        )
    policy: PolicyCfg = PolicyCfg()

@configclass
class StudentObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ProprioObservationsCfg(ObsGroup):
        """Observations for proprioception group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ExteroObservationsCfg(ObsGroup):
        """Observations for exteroception group."""

        # observation terms (order preserved)
        lidar_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("lidar_sensor")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ImageObservationsCfg(ObsGroup):
        """Observations for image group."""

        # observation terms (order preserved)
        head_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("head_camera"), "data_type": "rgb", "normalize": False,},
        )
        head_depth = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("head_camera"), "data_type": "depth"},
        )

        ee_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("ee_camera"), "data_type": "rgb", "normalize": False,},
        )
        ee_depth = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("ee_camera"), "data_type": "depth"},
        )

        ee_dual_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("ee_dual_camera"), "data_type": "rgb", "normalize": False,},
        )
        ee_dual_depth = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("ee_dual_camera"), "data_type": "depth"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    proprio: ProprioObservationsCfg = ProprioObservationsCfg()
    extero: ExteroObservationsCfg = ExteroObservationsCfg()
    image: ImageObservationsCfg = ImageObservationsCfg()

@configclass
class StudentObservationsCfg_TRAIN(StudentObservationsCfg):
    @configclass
    class PrivilegedObservationsCfg(ObsGroup):
        extreme_parkour_observations = ObsTerm(
            func=AtecTeacherObservations,
            params={            
            "asset_cfg":SceneEntityCfg("robot"),
            }
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DeltaYawOkCfg(ObsGroup):
        delta_yaw_ok = ObsTerm(
            func=observations.obervation_delta_yaw_ok,
            params={
                "asset_cfg":SceneEntityCfg("robot"),
                "parkour_name":'base_parkour',
                "threshold": 0.5
            },
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DepthCameraCfg(ObsGroup):
        depth_cam = ObsTerm(
            func=observations.image_features,
            params={
            "sensor_cfg":SceneEntityCfg("depth_camera"),
            "resize": (58, 87),
            "buffer_len": 2,
            "debug_vis":True
            },
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    teacher: PrivilegedObservationsCfg = PrivilegedObservationsCfg()
    delta_yaw_ok: DeltaYawOkCfg = DeltaYawOkCfg()
    depth_camera: DepthCameraCfg = DepthCameraCfg()


@configclass
class StudentRewardsCfg:
    reward_collision = RewTerm(
        func=rewards.reward_collision, 
        weight=-0., 
        params={
            "sensor_cfg":SceneEntityCfg("contact_forces", body_names=["base_link",".*_calf",".*_thigh"]),
        },
    )
    

@configclass
class TeacherRewardsCfg:
    """Reward terms for the MDP.
    ['base', 
    'FL_hip', 
    'FL_thigh', 
    'FL_calf', 
    'FL_foot', 
    'FR_hip', 
    'FR_thigh', 
    'FR_calf', 
    'FR_foot', 
    'Head_upper', 
    'Head_lower', 
    'RL_hip', 
    'RL_thigh', 
    'RL_calf', 
    'RL_foot', 
    'RR_hip', 
    'RR_thigh', 
    'RR_calf',
    'RR_foot']
    """
# Available Body strings: 
    reward_collision = RewTerm(
        func=rewards.reward_collision, 
        weight=-10., 
        params={
            "sensor_cfg":SceneEntityCfg("contact_forces", body_names=["base_link",".*_calf",".*_thigh"]),
        },
    )
    reward_feet_edge = RewTerm(
        func=rewards.reward_feet_edge, 
        weight=-1.0, 
        params={
            "asset_cfg":SceneEntityCfg(name="robot", body_names=["FL_foot","FR_foot","RL_foot","RR_foot"]),
            "sensor_cfg":SceneEntityCfg(name="contact_forces", body_names=".*_foot"),
            "parkour_name":'base_parkour',
        },
    )
    reward_torques = RewTerm(
        func=rewards.reward_torques, 
        weight=-0.00001, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    reward_dof_error = RewTerm(
        func=rewards.reward_dof_error, 
        weight=-0.04, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    reward_hip_pos = RewTerm(
        func=rewards.reward_hip_pos, 
        weight=-0.5, 
        params={
            "asset_cfg":SceneEntityCfg("robot", joint_names=".*_hip_joint"),
        },
    )
    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy, 
        weight=-0.05, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate, 
        weight=-0.1, 
        params={
          "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    reward_dof_acc = RewTerm(
        func=rewards.reward_dof_acc, 
        weight=-2.5e-7, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z, 
        weight=-1.0, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
            "parkour_name":'base_parkour',
        },
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation, 
        weight=-1.0, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
            "parkour_name":'base_parkour',
        },
    )
    reward_feet_stumble = RewTerm(
        func=rewards.reward_feet_stumble, 
        weight=-1.0, 
        params={
            "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    reward_tracking_goal_vel = RewTerm(
        func=rewards.reward_tracking_goal_vel, 
        weight=1.5, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
            "parkour_name":'base_parkour'
        },
    )
    reward_tracking_yaw = RewTerm(
        func=rewards.reward_tracking_yaw, 
        weight=0.5, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
            "parkour_name":'base_parkour'
        },
    )
    reward_delta_torques = RewTerm(
        func=rewards.reward_delta_torques, 
        weight=-1.0e-7, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    total_terminates = DoneTerm(
        func=terminations.terminate_episode, 
        time_out=True,
        params= {
            "asset_cfg":SceneEntityCfg("robot")
        },
    )
    
@configclass
class EventCfg:
    ### Modified origin events, plz see relative issue https://github.com/isaac-sim/IsaacLab/issues/1955
    """Configuration for events."""
    reset_root_state = EventTerm(
        func= events.reset_root_state,
        params = {'offset': 3.},
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func= reset_joints_by_scale, 
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
        mode="reset",
    )
    physics_material = None

    ## we don't use this event, If you use this, you will get a bad result
    # randomize_actuator_gains = EventTerm(
    #     func= events.randomize_actuator_gains,
    #     params={
    #         "asset_cfg" :SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.975, 1.025),  
    #         "damping_distribution_params": (0.975, 1.025),
    #         "operation": "scale",
    #         },
    #     mode="startup",
    # )
    randomize_rigid_body_mass = None
    randomize_rigid_body_com = None
    random_camera_position = EventTerm(
        func= events.random_camera_position,
        mode="startup",
        params={'sensor_cfg':SceneEntityCfg("depth_camera"),
                'rot_noise_range': {'pitch':(-5, 5)},
                'convention':'ros',
                },
    )
    push_by_setting_velocity = None
    base_external_force_torque = EventTerm(  # Okay
        func=apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

@configclass
class ActionsCfg:
    joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.25, 
        use_default_offset=True,
        action_delay_steps = [1, 1],
        delay_update_global_steps = 24 * 8000,
        history_length = 8,
        use_delay = True,
        clip = {'.*': (-4.8,4.8)}
        )
