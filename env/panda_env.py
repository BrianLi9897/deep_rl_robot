import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class PandaEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()

        # ===== Load MuJoCo model =====
        base_dir = os.path.dirname(__file__)
        xml_path = os.path.join(base_dir, "../assets/panda/scene.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_enabled = render
        self.viewer = None

        # ===== Actuator info =====
        self.n_act = self.model.nu
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.ctrl_mid = 0.5 * (self.ctrl_low + self.ctrl_high)
        self.ctrl_half_range = 0.5 * (self.ctrl_high - self.ctrl_low)

        # ===== Action space =====
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_act,),
            dtype=np.float32
        )

        # ===== Observation space =====
        obs_dim = self.model.nq + self.model.nv + 3 + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # ===== Target body (red ball) =====
        self.target_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target"
        )

        # ===== End-effector =====
        self.ee_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "hand"
        )

        # ===== Finger joints =====
        self.finger_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
        ]

        self.robot_base_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "link0"
        )

        # ===== Desk/cup placement params (must match scene.xml desk geom) =====
        self.table_center = np.array([0.0, 0.3], dtype=np.float64)
        self.table_half_size = np.array([0.5, 0.3], dtype=np.float64)
        self.table_top_z = 0.77  # desk_surface center z 0.75 + half-height 0.02
        self.cup_half_height = 0.04  # target_geom cylinder half-height in scene.xml
        self.cup_spawn_margin = 0.10
        self.cup_robot_clearance = 0.20

    # =========================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Randomize cup position on table surface each reset.
        # Use env RNG so seeding via reset(seed=...) is respected.
        span = self.table_half_size - self.cup_spawn_margin
        robot_xy = self.model.body_pos[self.robot_base_id][:2]

        # Rejection sampling keeps the cup from spawning hidden near the base.
        target_x, target_y = self.table_center[0], self.table_center[1]
        for _ in range(20):
            cand_x = self.table_center[0] + self.np_random.uniform(-span[0], span[0])
            cand_y = self.table_center[1] + self.np_random.uniform(-span[1], span[1])
            if np.linalg.norm(np.array([cand_x, cand_y]) - robot_xy) >= self.cup_robot_clearance:
                target_x, target_y = cand_x, cand_y
                break

        target_z = self.table_top_z + self.cup_half_height + 0.005
        
        target_pos = np.array([target_x, target_y, target_z])
        self.model.body_pos[self.target_id] = target_pos

        # Propagate changes to simulation
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    # =========================================================

    def step(self, action):
        # Normalize action → actuator control range
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        ctrl = self.ctrl_mid + action * self.ctrl_half_range
        self.data.ctrl[:] = np.clip(ctrl, self.ctrl_low, self.ctrl_high)

        # Step physics
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()

        done = False

        if self.render_enabled:
            self._render()

        return obs, reward, done, False, {}

    # =========================================================

    def _get_target(self):
        return self.model.body_pos[self.target_id].copy()

    # =========================================================

    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        target = self._get_target()
        gripper = self._get_gripper_opening()

        return np.concatenate([
            qpos,
            qvel,
            target,
            [gripper]
        ])

    # =========================================================

    def _get_gripper_opening(self):
        vals = [self.data.qpos[jid] for jid in self.finger_joint_ids]
        opening = np.mean(vals)

        # normalize (adjust range if needed)
        return np.clip(opening, 0.0, 0.04) / 0.04

    # =========================================================

    def _count_collisions(self, geom_names_exclude=None):
        """
        Count collisions excluding ground plane and internal Panda links.
        Returns number of active contacts.
        """
        if geom_names_exclude is None:
            geom_names_exclude = {"floor"}
        
        collision_count = 0
        for contact in self.data.contact:
            # Get geom IDs
            g1_id = contact.geom1
            g2_id = contact.geom2
            
            # Get geom names
            if g1_id >= 0 and g2_id >= 0:
                g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1_id)
                g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2_id)
                
                # Skip excluded geoms
                if g1_name in geom_names_exclude or g2_name in geom_names_exclude:
                    continue
                
                # Check if this is a collision we care about
                # (not internal Panda self-collisions that are expected)
                if g1_name and g2_name:
                    collision_count += 1
        
        return collision_count

    def _check_self_collision(self):
        """
        Check for problematic self-collisions (e.g., arm hitting gripper).
        Returns True if self-collision detected.
        """
        # Bodies that should not collide with each other
        arm_bodies = {"link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7"}
        gripper_bodies = {"hand", "left_finger", "right_finger"}
        
        for contact in self.data.contact:
            g1_id = contact.geom1
            g2_id = contact.geom2
            
            if g1_id >= 0 and g2_id >= 0:
                # Get body IDs from geom IDs using model's geom_bodyid array
                b1_id = self.model.geom_bodyid[g1_id]
                b2_id = self.model.geom_bodyid[g2_id]
                
                g1_body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b1_id)
                g2_body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b2_id)
                
                if g1_body and g2_body:
                    # Check for gripper hitting arm inappropriately
                    if (g1_body in gripper_bodies and g2_body in arm_bodies) or \
                       (g1_body in arm_bodies and g2_body in gripper_bodies):
                        return True
        
        return False

    # =========================================================

    def _compute_reward(self):
        ee_pos = self.data.xpos[self.ee_id]
        target = self._get_target()

        dist = np.linalg.norm(ee_pos - target)

        # Main reaching reward (distance to target)
        reward = -1.0 * dist

        # Intermediate success bonuses for getting closer
        if dist < 0.20:
            reward += 0.5  # Bonus for being within 20cm
        if dist < 0.10:
            reward += 1.0  # Bonus for being within 10cm

        # Final success bonus
        if dist < 0.05:
            reward += 3.0

            # Encourage closing gripper near target
            gripper = self._get_gripper_opening()
            reward += 0.2 * (1 - gripper)

        # Light smooth control penalty (encourage efficient motion, not jerky)
        reward -= 0.0001 * np.linalg.norm(self.data.ctrl)

        # Reduced velocity penalty (allow faster reaching motion)
        # Only applies if moving too fast (safety constraint)
        ee_vel = self.data.cvel[self.ee_id][:3]
        vel_norm = np.linalg.norm(ee_vel)
        if vel_norm > 1.0:  # Only penalize if moving faster than 1 m/s
            reward -= 0.001 * (vel_norm - 1.0)

        # Obstacle collision penalty (keep it modest so reaching is still prioritized)
        collision_count = self._count_collisions()
        reward -= 0.1 * collision_count

        # Self-collision penalty
        if self._check_self_collision():
            reward -= 0.5

        return reward

    # =========================================================

    def _render(self):
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()