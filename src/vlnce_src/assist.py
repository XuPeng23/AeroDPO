from src.vlnce_src.env_uav import RGB_FOLDER, DEPTH_FOLDER
from utils.logger import logger
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy


class Assist:
    def __init__(self, always_help=False, use_gt=False):
        self.always_help = always_help
        self.use_gt = use_gt
        
        self.safe_distance = 12.0           # front-camera safety distance
        self.safe_down = 5.0                # down-camera safety altitude
        
        self.k_att = 1.0                    # target attraction gain
        self.k_rep = 10.0                   # front repulsion gain
        self.k_rep_down = 10.0              # down repulsion gain
        
        self.max_fwd = 5.0
        self.max_yaw = 1.1
        self.max_down = 5.0
        self.fov_rad = np.radians(90.0)
        
        self._grid_initialized = False
        self.theta_h = None
        self.theta_v = None

    def _init_tensor_grid(self, h, w):
        """Precompute the camera angular grid used for 3D repulsion decomposition."""
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        self.theta_h = (x_grid - w/2) / w * self.fov_rad
        self.theta_v = (y_grid - h/2) / h * self.fov_rad
        self._grid_initialized = True

    def get_expert_action(self, current_obs, target_pos, rejected_action, collision_source=""):
        """
        APF (artificial potential field) reference action with a collision-direction filter.
        """
        chosen = copy.deepcopy(rejected_action)

        # ==========================================
        # 1. Run the APF to compute the omniscient reference action
        # ==========================================
        state = current_obs['sensors']['state']
        pos = np.array(state['position'])
        q = state['orientation']
        quat = [q.get('x',0), q.get('y',0), q.get('z',0), q.get('w',1)] if isinstance(q, dict) else list(q)
            
        # A. Target attraction
        vec_world = np.array(target_pos) - pos
        dist_to_target = np.linalg.norm(vec_world)
        vec_body = R.from_quat(quat).inv().apply(vec_world)
        f_att = (vec_body / dist_to_target) * self.k_att if dist_to_target > 1e-3 else np.zeros(3)

        # B. Front 3D repulsion integration (distinguishes going around vs. climbing over obstacles)
        depth_front = current_obs['depth'][0].astype(np.float32) / 255.0 * 100.0
        if not self._grid_initialized:
            self._init_tensor_grid(depth_front.shape[0], depth_front.shape[1])
            
        mask_front = depth_front < self.safe_distance
        f_rep_front = np.zeros(3)
        if np.any(mask_front):
            dists = np.clip(depth_front[mask_front], 0.1, self.safe_distance)
            mags = self.k_rep * ((self.safe_distance - dists) / self.safe_distance)  # linear
            rep_x = np.sum(-mags * np.cos(self.theta_v[mask_front]) * np.cos(self.theta_h[mask_front]))
            rep_y = np.sum(-mags * np.cos(self.theta_v[mask_front]) * np.sin(self.theta_h[mask_front]))
            rep_z = np.sum(-mags * np.sin(self.theta_v[mask_front]))
            f_rep_front = np.array([rep_x, rep_y, rep_z]) / np.sum(mask_front)

        # C. Down-view spring repulsion
        depth_down = current_obs['depth'][1].astype(np.float32) / 255.0 * 100.0
        min_down = max(np.min(depth_down[int(depth_down.shape[0]*0.3):int(depth_down.shape[0]*0.7), 
                                         int(depth_down.shape[1]*0.3):int(depth_down.shape[1]*0.7)]), 0.1)
        if min_down < self.safe_down:
            mag_down = self.k_rep_down * ((self.safe_down - min_down) / self.safe_down)
            f_rep_down = np.array([0.0, 0.0, -mag_down])
        else:
            f_rep_down = np.zeros(3)

        f_total = f_att + f_rep_front + f_rep_down

        yaw_x_base = max(f_total[0], 0.5)
        apf_yaw = np.clip(np.arctan2(f_total[1], yaw_x_base), -self.max_yaw, self.max_yaw)
        apf_down = np.clip(f_total[2], -self.max_down, self.max_down)
        speed_scale = np.clip((np.min(depth_front) - 1.0) / 3.0, 0.1, 1.0) if np.any(mask_front) else 1.0
        apf_fwd = np.clip(np.linalg.norm(f_total[:2]) * speed_scale, 0.0, self.max_fwd)

        apf_raw_action = {
            'fwd': float(apf_fwd),
            'down': float(apf_down),
            'yaw': float(apf_yaw)
        }

        # ==========================================
        # 2. Directional dimension filter: override only the dimension(s) flagged
        # by the collision alarm; keep the model's intent in the other dimensions.
        # ==========================================
        
        if collision_source == 'downcamera_depth':
            # Ground proximity: adopt only the APF altitude-climb command,
            # keeping the model's heading intent.
            chosen['down'] = apf_down
        elif collision_source in ['leftcamera_depth', 'rightcamera_depth']:
            # Lateral contact: adopt only the APF avoidance yaw,
            # keeping the model's altitude and forward intent.
            chosen['yaw'] = apf_yaw
        elif collision_source == 'frontcamera_depth':
            # Head-on collision: single-dimension escape (winner takes all)
            
            # Compare the normalized deviation between the APF suggestion and the model
            # action to decide which dimension is more urgent to correct.
            norm_delta_down = abs(apf_down - chosen['down']) / self.max_down
            norm_delta_yaw = abs(apf_yaw - chosen['yaw']) / self.max_yaw

            if norm_delta_down > norm_delta_yaw:
                chosen['down'] = apf_down
                
                if (chosen['fwd'] - apf_fwd) > 2.0:
                    chosen['fwd'] = apf_fwd
                    
            else:
                chosen['yaw'] = apf_yaw
        
        f_att_world = (vec_world / dist_to_target) * self.k_att if dist_to_target > 1e-3 else np.zeros(3)
        f_rep_body = f_rep_front + f_rep_down
        f_rep_world = R.from_quat(quat).apply(f_rep_body) 

        return {
            'chosen': chosen,
            'apf': apf_raw_action,
            'f_att_world': f_att_world.tolist(),
            'f_rep_world': f_rep_world.tolist()
        }

    def check_collision_by_depth(self, episodes, current_observations, collisions, dones):
        
        collision_sources = [""] * len(episodes)
        
        for i, prev_episode in enumerate(episodes):
            collision_type = None
            if collisions[i]:
                collision_type = 'already'
                if not dones[i]:
                    dones[i] = True
                continue
            
            close_collision = False
            current_episode = current_observations[i]

            detected_camera = ""

            for cid, camera_name in enumerate(DEPTH_FOLDER):
                zero_cnt  = (current_episode[-1]['depth'][cid] <= 1).sum()
                if zero_cnt > 0.1 * current_episode[-1]['depth'][cid].size:
                    close_collision = True
                    logger.info(f"Collision detected: {camera_name} depth")

                    if not detected_camera: 
                        detected_camera = camera_name

            if close_collision:
                collision_type = 'close'
            
            if collision_type is not None:
                logger.info(f"Collision type: {collision_type}")

            collisions[i] = close_collision
            collision_sources[i] = detected_camera

            if collisions[i] and not dones[i]:
                dones[i] = True

        return collisions, dones, collision_sources