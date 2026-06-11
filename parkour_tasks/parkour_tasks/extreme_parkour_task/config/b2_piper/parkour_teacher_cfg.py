import os
import numpy as np
from copy import deepcopy
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
from scipy.spatial.transform import Rotation as R
from isaaclab.utils import configclass

from parkour_tasks.extreme_parkour_task.config.go2.parkour_teacher_cfg import ParkourTeacherSceneCfg, UnitreeGo2TeacherParkourEnvCfg, UnitreeGo2TeacherParkourEnvCfg_PLAY, UnitreeGo2TeacherParkourEnvCfg_EVAL
PARKOUR_ISAACLAB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../parkour_isaaclab"))
B2_USD_PATH = os.path.join(PARKOUR_ISAACLAB_DIR, "assets", "robots", "b2", "b2.usd")
B2_PIPER_USD_PATH = os.path.join(PARKOUR_ISAACLAB_DIR, "assets", "robots", "b2", "b2_piper.usda")

UNITREE_B2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{B2_USD_PATH}",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            "arm_joint1": 0.0,
            "arm_joint2": 2.5,
            "arm_joint3": -2.5,
            "arm_joint4": 0.0,
            "arm_joint5": 0.0,
            "arm_joint6": 0.0,
            "arm_joint7": 0.0,
            "arm_joint8": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "M107-24-2": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_thigh_.*"],
            effort_limit_sim=200.0,
            velocity_limit_sim=23.0,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
            armature=0.01,
        ),
        "2": ImplicitActuatorCfg(
            joint_names_expr=[".*_calf_.*"],
            effort_limit_sim=320.0,
            velocity_limit_sim=14.0,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
            armature=0.01,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)

UNITREE_B2_PIPER_CFG = deepcopy(UNITREE_B2_CFG)
UNITREE_B2_PIPER_CFG.spawn.articulation_props.enabled_self_collisions = False
UNITREE_B2_PIPER_CFG.spawn.usd_path = str(B2_PIPER_USD_PATH)
UNITREE_B2_PIPER_CFG.actuators["arms"] = ImplicitActuatorCfg(
    joint_names_expr=["arm_joint.*"],
    effort_limit_sim=100.0,
    velocity_limit_sim=100.0,
    stiffness=80.0,
    damping=4.0,
    friction=0.01,
    armature=0.01,
)

@configclass
class B2PiperTeacherSceneCfg(ParkourTeacherSceneCfg):
    robot = UNITREE_B2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.1, # from Task D
        pattern_cfg=patterns.LidarPatternCfg(
            vertical_fov_range=(-20.0, 20.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
            channels=16,
        ),
        max_distance=10.0,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    
    def __post_init__(self):
        super().__post_init__()
        # We don't want to use ParkourDCMotorCfg from the go2 setup for B2 piper as the B2
        # limits are different. However, ParkourTeacherSceneCfg calls super().__post_init__()
        # which overwrites self.robot.actuators['base_legs']. We must remove it to avoid errors
        # as it relies on Go2 joint names, or redefine it. We'll simply delete it and rely on
        # the built-in actuators.
        if 'base_legs' in self.robot.actuators:
            del self.robot.actuators['base_legs']

from parkour_tasks.extreme_parkour_task.config.go2.parkour_mdp_cfg import TeacherRewardsCfg, EventCfg, StudentRewardsCfg
from parkour_isaaclab.envs.mdp.parkour_actions import DelayedJointPositionActionCfg
from isaaclab.managers import SceneEntityCfg

@configclass
class B2PiperTeacherRewardsCfg(TeacherRewardsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.reward_collision.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=["base_link",".*_calf",".*_thigh","arm_.*","gripper.*"])

@configclass
class B2PiperEventCfg(EventCfg):
    def __post_init__(self):
        super().__post_init__()
        self.randomize_rigid_body_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.randomize_rigid_body_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.physics_material.params["friction_range"] = (1.0, 1.0)

from parkour_isaaclab.envs.mdp.observations import CompetitionTaskDObservations
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from parkour_tasks.extreme_parkour_task.config.go2.parkour_mdp_cfg import TeacherObservationsCfg

@configclass
class B2PiperTeacherObservationsCfg(TeacherObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        extreme_parkour_observations = ObsTerm(
            func=CompetitionTaskDObservations,
            params={            
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
    policy: PolicyCfg = PolicyCfg()

@configclass
class B2PiperActionsCfg:
    joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"], 
        scale=0.25, 
        use_default_offset=True,
        action_delay_steps = [1, 1],
        delay_update_global_steps = 24 * 8000,
        history_length = 8,
        use_delay = True,
        clip = {'.*': (-4.8,4.8)}
    )

@configclass
class B2PiperTeacherParkourEnvCfg(UnitreeGo2TeacherParkourEnvCfg):
    scene: B2PiperTeacherSceneCfg = B2PiperTeacherSceneCfg(num_envs=6144, env_spacing=1.)
    rewards: B2PiperTeacherRewardsCfg = B2PiperTeacherRewardsCfg()
    events: B2PiperEventCfg = B2PiperEventCfg()
    observations: B2PiperTeacherObservationsCfg = B2PiperTeacherObservationsCfg()
    actions: B2PiperActionsCfg = B2PiperActionsCfg()

@configclass
class B2PiperTeacherParkourEnvCfg_EVAL(UnitreeGo2TeacherParkourEnvCfg_EVAL):
    scene: B2PiperTeacherSceneCfg = B2PiperTeacherSceneCfg(num_envs=256, env_spacing=1.)
    rewards: B2PiperTeacherRewardsCfg = B2PiperTeacherRewardsCfg()
    events: B2PiperEventCfg = B2PiperEventCfg()
    observations: B2PiperTeacherObservationsCfg = B2PiperTeacherObservationsCfg()
    actions: B2PiperActionsCfg = B2PiperActionsCfg()

@configclass
class B2PiperTeacherParkourEnvCfg_PLAY(UnitreeGo2TeacherParkourEnvCfg_PLAY):
    scene: B2PiperTeacherSceneCfg = B2PiperTeacherSceneCfg(num_envs=16, env_spacing=1.)
    rewards: B2PiperTeacherRewardsCfg = B2PiperTeacherRewardsCfg()
    events: B2PiperEventCfg = B2PiperEventCfg()
    observations: B2PiperTeacherObservationsCfg = B2PiperTeacherObservationsCfg()
    actions: B2PiperActionsCfg = B2PiperActionsCfg()

