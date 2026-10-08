import math
import random
import time
import cv2
import numpy as np
import pybullet as p
import pybullet_data
import pyttsx3

# ==========================================
# 0. SIMPLE DIRECT SPEECH FUNCTION
# ==========================================


def speak_status(phrase):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
        voices = engine.getProperty("voices")
        for v in voices:
            if "female" in v.name.lower() or "zira" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        engine.say(phrase)
        engine.runAndWait()
    except Exception:
        pass


# ==========================================
# 1. SIMULATION & TERRAIN SETUP
# ==========================================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

mesh_scale = [0.1, 0.1, 24.0]
terrain_shape = p.createCollisionShape(
    shapeType=p.GEOM_HEIGHTFIELD,
    fileName="heightmaps/wm_height_out.png",
    meshScale=mesh_scale,
    heightfieldTextureScaling=128,
)
terrain_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=terrain_shape)
p.changeVisualShape(terrain_body, -1, rgbaColor=[0.22, 0.45, 0.20, 1.0])

random.seed(42)
num_trees = 60
trunk_height, trunk_radius, foliage_radius = 1.2, 0.15, 0.6
tree_positions = []

trunk_shape = p.createCollisionShape(
    p.GEOM_CYLINDER, radius=trunk_radius, height=trunk_height
)
trunk_visual = p.createVisualShape(
    p.GEOM_CYLINDER,
    radius=trunk_radius,
    length=trunk_height,
    rgbaColor=[0.35, 0.20, 0.05, 1.0],
)
foliage_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=foliage_radius)
foliage_visual = p.createVisualShape(
    p.GEOM_SPHERE, radius=foliage_radius, rgbaColor=[0.15, 0.48, 0.15, 1.0]
)

for _ in range(num_trees):
    tx, ty = random.uniform(-8.5, 8.5), random.uniform(-8.5, 8.5)
    ray_res = p.rayTest([tx, ty, 50], [tx, ty, -10])
    if ray_res[0][0] == terrain_body and ray_res[0][4][2] > 0.85:
        tz = ray_res[0][3][2]
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=trunk_shape,
            baseVisualShapeIndex=trunk_visual,
            basePosition=[tx, ty, tz + trunk_height / 2],
            baseOrientation=[0, 0, 0, 1],
            linkMasses=[0.0],
            linkCollisionShapeIndices=[foliage_shape],
            linkVisualShapeIndices=[foliage_visual],
            linkPositions=[
                [0, 0, trunk_height / 2 + foliage_radius * 0.5]
            ],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]],
        )
        tree_positions.append((tx, ty))

drone_shape = p.createCollisionShape(
    p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.04]
)
drone_visual = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.04], rgbaColor=[0.85, 0.15, 0.15, 1.0]
)

start_pos = [-8.0, -8.0, 15.0]
drone_body = p.createMultiBody(
    1.0, drone_shape, drone_visual, start_pos, [0, 0, 0, 1]
)

p.resetDebugVisualizerCamera(26.0, 45.0, -35.0, [0, 0, 4])

# ==========================================
# 2. SENSORS & THREAT SIMULATOR
# ==========================================
REF_LAT, REF_LON = 22.3089, 39.1042
LAT_DEG_TO_METERS = 111000.0
LON_DEG_TO_METERS = 111000.0 * math.cos(math.radians(REF_LAT))

GPS_STATE_NORMAL = 0
GPS_STATE_JAMMED = 1
GPS_STATE_SPOOFED = 2


def get_simulated_gps(true_pos, state, sim_time):
    if state == GPS_STATE_JAMMED:
        return None, None, None, None
    elif state == GPS_STATE_SPOOFED:
        spoof_x = true_pos[0] + 12.0 * math.sin(sim_time * 0.4)
        spoof_y = true_pos[1] + 12.0 * math.cos(sim_time * 0.4)
        spoof_z = true_pos[2] + 4.0
        lat = REF_LAT + (spoof_y / LAT_DEG_TO_METERS)
        lon = REF_LON + (spoof_x / LON_DEG_TO_METERS)
        return lat, lon, spoof_z, np.array([spoof_x, spoof_y, spoof_z])
    else:
        nx = true_pos[0] + np.random.normal(0, 0.15)
        ny = true_pos[1] + np.random.normal(0, 0.15)
        nz = true_pos[2] + np.random.normal(0, 0.08)
        lat = REF_LAT + (ny / LAT_DEG_TO_METERS)
        lon = REF_LON + (nx / LON_DEG_TO_METERS)
        return lat, lon, nz, np.array([nx, ny, nz])


prev_vel = np.array([0.0, 0.0, 0.0])


def get_simulated_imu(drone_id, dt=0.033):
    global prev_vel
    lin_vel, ang_vel = p.getBaseVelocity(drone_id)
    _, orn = p.getBasePositionAndOrientation(drone_id)
    accel_world = (np.array(lin_vel) - prev_vel) / dt + np.array(
        [0.0, 0.0, 9.81]
    )
    prev_vel = np.array(lin_vel)
    rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    accel_body = rot_mat.T @ accel_world + np.random.normal(0, 0.05, 3)
    gyro_body = rot_mat.T @ np.array(ang_vel) + np.random.normal(0, 0.01, 3)
    roll, pitch, yaw = p.getEulerFromQuaternion(orn)
    return accel_body, gyro_body, (roll, pitch, yaw)


NUM_LIDAR_RAYS = 120
LIDAR_MAX_RANGE = 40.0


def get_simulated_lidar(drone_pos, drone_orn):
    rot_mat = np.array(p.getMatrixFromQuaternion(drone_orn)).reshape(3, 3)
    starts, ends = [], []
    angles = np.linspace(0, 2 * np.pi, NUM_LIDAR_RAYS, endpoint=False)
    for a in angles:
        dir_local = np.array([math.cos(a) * 0.8, math.sin(a) * 0.8, -0.6])
        dir_world = rot_mat @ dir_local
        starts.append(np.array(drone_pos))
        ends.append(np.array(drone_pos) + dir_world * LIDAR_MAX_RANGE)

    results = p.rayTestBatch(starts, ends)
    polar_data = []
    distances_only = []
    for i, res in enumerate(results):
        dist = res[2] * LIDAR_MAX_RANGE
        polar_data.append((angles[i], dist, res[2] < 1.0))
        distances_only.append(dist)
    return polar_data, np.array(distances_only)


# ==========================================
# 3. SAFETY & GEOFENCE ENGINES
# ==========================================
GEOFENCE_LIMIT = 10.0
MIN_GROUND_CLEARANCE = 8.0


def get_terrain_height_below(x, y):
    ray = p.rayTest([x, y, 60.0], [x, y, -20.0])
    if ray[0][0] == terrain_body:
        return ray[0][3][2]
    return 0.0


def compute_obstacle_avoidance_vector(lidar_polar, current_yaw):
    repulsion_x, repulsion_y = 0.0, 0.0
    for local_angle, dist, has_hit in lidar_polar:
        if has_hit and dist < 4.0:
            global_angle = local_angle + current_yaw
            repulsion_force = ((4.0 - dist) / 4.0) ** 2
            repulsion_x -= math.cos(global_angle) * repulsion_force * 0.3
            repulsion_y -= math.sin(global_angle) * repulsion_force * 0.3
    return repulsion_x, repulsion_y


def enforce_geofence_smooth(pos_x, pos_y):
    fence_margin = 2.0
    max_dist = GEOFENCE_LIMIT - fence_margin
    fence_x, fence_y = 0.0, 0.0
    if abs(pos_x) > max_dist:
        overlap = abs(pos_x) - max_dist
        fence_x = -math.copysign((overlap / fence_margin) ** 2, pos_x) * 1.0
    if abs(pos_y) > max_dist:
        overlap = abs(pos_y) - max_dist
        fence_y = -math.copysign((overlap / fence_margin) ** 2, pos_y) * 1.0
    return fence_x, fence_y


# ==========================================
# 4. ONBOARD SPATIAL MEMORY BANK
# ==========================================
spatial_memory_nodes = []
last_memory_save_pos = np.array([999.0, 999.0, 999.0])


def save_memory_node(pos, yaw, lidar_sig, mean_depth):
    node_id = len(spatial_memory_nodes) + 1
    node = {
        "id": node_id,
        "pos": pos.copy(),
        "yaw": yaw,
        "lidar_signature": lidar_sig.copy(),
        "mean_depth": mean_depth,
    }
    spatial_memory_nodes.append(node)
    return node_id


def query_memory_bank(live_lidar_sig, live_mean_depth):
    if len(spatial_memory_nodes) == 0:
        return None, 999.0
    best_match_node = None
    min_mae_error = 999.0
    for node in spatial_memory_nodes:
        lidar_diff = np.mean(
            np.abs(live_lidar_sig - node["lidar_signature"])
        )
        depth_diff = abs(live_mean_depth - node["mean_depth"])
        composite_score = lidar_diff + (depth_diff * 0.5)
        if composite_score < min_mae_error:
            min_mae_error = composite_score
            best_match_node = node
    if min_mae_error < 5.0:
        return best_match_node, min_mae_error
    else:
        return None, min_mae_error


# ==========================================
# 5. 2D MINIMAP & FINAL ACCURACY GRAPH RENDERER
# ==========================================
map_size = 450
world_min, world_max = -12.0, 12.0
path_history = []
accuracy_history = []  # Stores tuples of (time, error, state)


def world_to_map_px(x, y):
    px = int((x - world_min) / (world_max - world_min) * map_size)
    py = int((world_max - y) / (world_max - world_min) * map_size)
    return px, py


def render_mario_kart_minimap(
    true_pos, est_pos, waypoints, current_wp_idx, memory_nodes, trees, yaw, phase_str
):
    minimap = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    minimap[:] = (20, 25, 20)

    for g in range(-10, 11, 5):
        gx1, gy1 = world_to_map_px(g, -12)
        gx2, gy2 = world_to_map_px(g, 12)
        cv2.line(minimap, (gx1, gy1), (gx2, gy2), (35, 45, 35), 1)
        gx1, gy1 = world_to_map_px(-12, g)
        gx2, gy2 = world_to_map_px(12, g)
        cv2.line(minimap, (gx1, gy1), (gx2, gy2), (35, 45, 35), 1)

    bx1, by1 = world_to_map_px(-GEOFENCE_LIMIT, GEOFENCE_LIMIT)
    bx2, by2 = world_to_map_px(GEOFENCE_LIMIT, -GEOFENCE_LIMIT)
    cv2.rectangle(minimap, (bx1, by1), (bx2, by2), (0, 0, 180), 2)

    for tx, ty in trees:
        px, py = world_to_map_px(tx, ty)
        cv2.circle(minimap, (px, py), 4, (10, 100, 10), -1)

    for i in range(len(waypoints) - 1):
        pt1 = world_to_map_px(waypoints[i][0], waypoints[i][1])
        pt2 = world_to_map_px(waypoints[i + 1][0], waypoints[i + 1][1])
        cv2.line(minimap, pt1, pt2, (180, 120, 0), 2)

    for idx, wp in enumerate(waypoints):
        wpx, wpy = world_to_map_px(wp[0], wp[1])
        color = (0, 255, 255) if idx == current_wp_idx else (150, 150, 150)
        radius = 7 if idx == current_wp_idx else 4
        cv2.circle(minimap, (wpx, wpy), radius, color, -1)
        cv2.putText(
            minimap,
            str(idx + 1),
            (wpx + 5, wpy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )

    for node in memory_nodes:
        mx, my = world_to_map_px(node["pos"][0], node["pos"][1])
        cv2.circle(minimap, (mx, my), 3, (0, 255, 0), -1)

    for i in range(1, len(path_history)):
        p1 = world_to_map_px(path_history[i - 1][0], path_history[i - 1][1])
        p2 = world_to_map_px(path_history[i][0], path_history[i][1])
        cv2.line(minimap, p1, p2, (0, 0, 255), 1)

    ex, ey = world_to_map_px(est_pos[0], est_pos[1])
    cv2.circle(minimap, (ex, ey), 5, (0, 255, 255), 1)

    dx, dy = world_to_map_px(true_pos[0], true_pos[1])
    arrow_len = 14
    hx = int(dx + arrow_len * math.cos(yaw))
    hy = int(dy - arrow_len * math.sin(yaw))

    cv2.arrowedLine(minimap, (dx, dy), (hx, hy), (0, 0, 255), 3, tipLength=0.4)
    cv2.circle(minimap, (dx, dy), 6, (0, 0, 255), -1)

    cv2.putText(
        minimap,
        f"PHASE: {phase_str}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        minimap,
        "Red = Drone | Yellow = Estimate | Green = Memory",
        (10, 440),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (200, 200, 200),
        1,
    )
    return minimap


def generate_final_accuracy_report(history):
    graph_w, graph_h = 600, 400
    report_img = np.zeros((graph_h, graph_w, 3), dtype=np.uint8)
    report_img[:] = (20, 25, 20)

    cv2.putText(
        report_img,
        "MISSION COMPLETE: NAVIGATION ACCURACY REPORT",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )

    if len(history) < 2:
        return report_img

    times = [h[0] for h in history]
    errors = [h[1] for h in history]
    max_t = max(times) if times else 1.0
    max_e = max(max(errors), 15.0)

    # Draw grid & axes
    cv2.line(report_img, (50, graph_h - 60), (graph_w - 30, graph_h - 60), (200, 200, 200), 1)
    cv2.line(report_img, (50, 50), (50, graph_h - 60), (200, 200, 200), 1)

    cv2.putText(report_img, "Drift Error (m)", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(report_img, "Mission Timeline (s)", graph_w - 140, graph_h - 35, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    for i in range(1, len(history)):
        t1, err1, state1 = history[i - 1]
        t2, err2, state2 = history[i]

        x1 = int(50 + (t1 / max_t) * (graph_w - 90))
        x2 = int(50 + (t2 / max_t) * (graph_w - 90))

        y1 = int(graph_h - 60 - min(1.0, err1 / max_e) * (graph_h - 120))
        y2 = int(graph_h - 60 - min(1.0, err2 / max_e) * (graph_h - 120))

        color = (0, 255, 0) if state1 == 0 else ((0, 0, 255) if state1 == 1 else (0, 165, 255))
        cv2.line(report_img, (x1, y1), (x2, y2), color, 2)

    # Summary statistics
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    cv2.putText(report_img, f"Avg Drift: {avg_error:.2f}m | Peak Drift: {max_error:.2f}m", (50, graph_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return report_img


# ==========================================
# 6. MAIN CONTROL LOOP
# ==========================================
waypoints = [
    [-8.0, -8.0, 15.0],
    [-8.0, 8.0, 15.0],
    [-4.0, 8.0, 15.0],
    [-4.0, -8.0, 15.0],
    [0.0, -8.0, 15.0],
    [0.0, 8.0, 15.0],
    [4.0, 8.0, 15.0],
    [4.0, -8.0, 15.0],
    [8.0, -8.0, 15.0],
    [8.0, 8.0, 15.0],
]

current_wp_idx = 0
cruise_speed = 0.12
cam_w, cam_h = 360, 270

true_pos = np.array(start_pos, dtype=np.float64)
est_pos = np.array(start_pos, dtype=np.float64)
accumulated_drift_vector = np.array([0.0, 0.0, 0.0])
current_yaw = 0.0

sim_start_time = time.time()
last_map_match_time = time.time()
map_match_status = "INITIALIZING MEMORY..."

mission_phase = 1
gps_threat_state = GPS_STATE_NORMAL
last_spoken_threat_state = -99
mission_complete = False
report_displayed = False

speak_status("Phase 1 initialized. Surveying region to record full spatial memory map.")

while True:
    dt = 0.033
    sim_elapsed = time.time() - sim_start_time

    path_history.append((true_pos[0], true_pos[1]))
    if len(path_history) > 500:
        path_history.pop(0)

    # --- Phase & Threat Simulation Schedule ---
    if mission_phase == 1:
        gps_threat_state = GPS_STATE_NORMAL
        phase_str = "1. SURVEY FLIGHT (RECORDING MAP)"
    else:
        if mission_complete:
            phase_str = "MISSION COMPLETE (HOLDING POSITION)"
            gps_threat_state = GPS_STATE_NORMAL
        else:
            phase_str = "2. OPERATIONAL MISSION (THREATS ACTIVE)"
            threat_cycle = int(sim_elapsed) % 30
            if threat_cycle < 10:
                gps_threat_state = GPS_STATE_NORMAL
            elif threat_cycle < 20:
                gps_threat_state = GPS_STATE_JAMMED
            else:
                gps_threat_state = GPS_STATE_SPOOFED

    # --- VOICE TRIGGER IF-ELSE ---
    if mission_phase == 2 and not mission_complete:
        if gps_threat_state != last_spoken_threat_state:
            if gps_threat_state == GPS_STATE_NORMAL:
                speak_status("GPS signal normalized. Resuming nominal path.")
            elif gps_threat_state == GPS_STATE_JAMMED:
                speak_status("Warning! GPS jamming detected. Switching to onboard memory.")
            elif gps_threat_state == GPS_STATE_SPOOFED:
                speak_status("Alert! GPS spoofing detected. Rejecting false coordinates.")
            last_spoken_threat_state = gps_threat_state

    # Read Hardware Sensors
    imu_accel, imu_gyro, _ = get_simulated_imu(drone_body, dt)
    gps_lat, gps_lon, gps_alt, raw_gps_pos = get_simulated_gps(true_pos, gps_threat_state, sim_elapsed)
    lidar_polar, lidar_sig = get_simulated_lidar(true_pos, p.getQuaternionFromEuler([0, 0, current_yaw]))

    # Render Camera
    rot_matrix = p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, current_yaw]))
    forward_vec = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
    up_vec = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]

    cam_pos = [true_pos[0] + 0.3 * forward_vec[0], true_pos[1] + 0.3 * forward_vec[1], true_pos[2] + 0.05]
    cam_target = [cam_pos[0] + 2.5 * forward_vec[0], cam_pos[1] + 2.5 * forward_vec[1], cam_pos[2] + 2.5 * forward_vec[2] - 1.8]

    view_mat = p.computeViewMatrix(cam_pos, cam_target, up_vec)
    proj_mat = p.computeProjectionMatrixFOV(85, cam_w / cam_h, 0.1, 100.0)

    img_data = p.getCameraImage(cam_w, cam_h, view_mat, proj_mat, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    depth_buf = np.reshape(img_data[3], (cam_h, cam_w))
    depth_m = (100.0 * 0.1) / (100.0 - (100.0 - 0.1) * depth_buf + 1e-6)
    live_mean_depth = float(np.mean(depth_m))

    # --- Position Estimation & Memory Management ---
    if gps_threat_state == GPS_STATE_NORMAL:
        est_pos = raw_gps_pos.copy()
        accumulated_drift_vector = np.array([0.0, 0.0, 0.0])

        if np.linalg.norm(true_pos - last_memory_save_pos) >= 1.5:
            node_id = save_memory_node(true_pos, current_yaw, lidar_sig, live_mean_depth)
            last_memory_save_pos = true_pos.copy()
            map_match_status = f"SAVED MEMORY NODE #{node_id}"
        else:
            map_match_status = f"SAVING MAP ({len(spatial_memory_nodes)} STORED)"
    else:
        drift_bias = np.array([math.sin(sim_elapsed * 1.5) * 0.03, math.cos(sim_elapsed * 1.5) * 0.03, 0.01])
        accumulated_drift_vector += drift_bias
        est_pos = true_pos + accumulated_drift_vector

        if time.time() - last_map_match_time > 2.0:
            matched_node, mae_score = query_memory_bank(lidar_sig, live_mean_depth)
            if matched_node is not None:
                est_pos = matched_node["pos"].copy()
                accumulated_drift_vector = est_pos - true_pos
                map_match_status = f"MATCHED NODE #{matched_node['id']}! DRIFT RESET"
                last_map_match_time = time.time()
            else:
                map_match_status = f"SEARCHING MEMORY... (MAE: {mae_score:.2f})"

    position_error_meters = np.linalg.norm(true_pos - est_pos)

    # Record data point for final graph
    accuracy_history.append((sim_elapsed, position_error_meters, gps_threat_state))

    # --- WAYPOINT NAVIGATION & HOVER LOCK ---
    if not mission_complete:
        target_wp = waypoints[current_wp_idx]
        wp_dx = target_wp[0] - true_pos[0]
        wp_dy = target_wp[1] - true_pos[1]
        wp_dist = math.sqrt(wp_dx**2 + wp_dy**2)

        if wp_dist < 1.6:
            current_wp_idx += 1
            if current_wp_idx >= len(waypoints):
                if mission_phase == 1:
                    mission_phase = 2
                    current_wp_idx = 0
                    speak_status("Phase 1 survey complete. Entering Phase 2 operational mission.")
                else:
                    mission_complete = True
                    speak_status("Mission complete. Holding final position.")
                    current_wp_idx = len(waypoints) - 1

        if not mission_complete:
            target_wp = waypoints[current_wp_idx]
            wp_dx = target_wp[0] - true_pos[0]
            wp_dy = target_wp[1] - true_pos[1]
            wp_dist = math.sqrt(wp_dx**2 + wp_dy**2)

            wp_dir_x = wp_dx / (wp_dist + 1e-6)
            wp_dir_y = wp_dy / (wp_dist + 1e-6)

            avoid_x, avoid_y = compute_obstacle_avoidance_vector(lidar_polar, current_yaw)
            fence_x, fence_y = enforce_geofence_smooth(true_pos[0], true_pos[1])

            combined_dx = wp_dir_x + avoid_x * 1.5 + fence_x * 2.0
            combined_dy = wp_dir_y + avoid_y * 1.5 + fence_y * 2.0

            target_yaw = math.atan2(combined_dy, combined_dx)
            yaw_diff = (target_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
            max_turn_rate = 0.05
            yaw_diff = max(-max_turn_rate, min(max_turn_rate, yaw_diff))
            current_yaw += yaw_diff

            true_pos[0] += cruise_speed * math.cos(current_yaw)
            true_pos[1] += cruise_speed * math.sin(current_yaw)

            ground_z = get_terrain_height_below(true_pos[0], true_pos[1])
            desired_altitude = max(target_wp[2], ground_z + MIN_GROUND_CLEARANCE)
            true_pos[2] += (desired_altitude - true_pos[2]) * 0.1

    true_pos[0] = np.clip(true_pos[0], -GEOFENCE_LIMIT, GEOFENCE_LIMIT)
    true_pos[1] = np.clip(true_pos[1], -GEOFENCE_LIMIT, GEOFENCE_LIMIT)

    drone_orn = p.getQuaternionFromEuler([0, 0, current_yaw])
    p.resetBasePositionAndOrientation(drone_body, true_pos, drone_orn)

    p.stepSimulation()

    # --- OpenCV HUD Rendering ---
    rgba = np.reshape(img_data[2], (cam_h, cam_w, 4))
    rgb_bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    geo_bgr = cv2.applyColorMap(np.clip((depth_m - 2.0) / 25.0 * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    thermal_bgr = cv2.applyColorMap(np.clip((27.0 - depth_m) / 25.0 * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    lidar_canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
    center = (cam_w // 2, cam_h // 2)
    for r_m in [10, 20, 30, 40]:
        cv2.circle(lidar_canvas, center, int((r_m / LIDAR_MAX_RANGE) * 110), (40, 40, 40), 1)

    for ang, dist, has_hit in lidar_polar:
        if has_hit:
            px = int(center[0] + (dist / LIDAR_MAX_RANGE) * 110 * math.cos(ang))
            py = int(center[1] + (dist / LIDAR_MAX_RANGE) * 110 * math.sin(ang))
            pt_col = (0, 0, 255) if dist < 4.0 else ((0, 255, 255) if dist < 15.0 else (0, 255, 0))
            cv2.circle(lidar_canvas, (px, py), 2, pt_col, -1)
    cv2.circle(lidar_canvas, center, 4, (255, 255, 255), -1)

    cv2.putText(rgb_bgr, "1. OPTICAL FLOW (CAMERA)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
    cv2.putText(geo_bgr, "2. DEPTH TERRAIN PROFILER", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.putText(thermal_bgr, "3. FLIR THERMAL CAMERA", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    cv2.putText(lidar_canvas, "4. 360 LIDAR RADAR", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

    grid = np.vstack((np.hstack((rgb_bgr, geo_bgr)), np.hstack((thermal_bgr, lidar_canvas))))

    banner = np.zeros((125, grid.shape[1], 3), dtype=np.uint8)
    state_str = "NORMAL (GPS OK)" if gps_threat_state == 0 else ("JAMMED (GPS LOSS)" if gps_threat_state == 1 else "SPOOFED (FALSE GPS)")
    state_col = (0, 255, 0) if gps_threat_state == 0 else ((0, 0, 255) if gps_threat_state == 1 else (0, 165, 255))

    cv2.putText(banner, f"GPS STATUS        : {state_str}", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, state_col, 2)
    cv2.putText(banner, f"MISSION PHASE     : {phase_str}", (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 2)
    cv2.putText(banner, f"TERRAIN CLEARANCE : GROUND {ground_z:.1f}m | ALTITUDE {true_pos[2]:.1f}m", (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
    cv2.putText(banner, f"GROUND TRUTH      : [{true_pos[0]:+.1f}, {true_pos[1]:+.1f}] | DRIFT ERROR: {position_error_meters:.2f}m", (15, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.putText(banner, f"MEMORY BANK       : {map_match_status}", (15, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

    full_hud = np.vstack((grid, banner))
    minimap_img = render_mario_kart_minimap(true_pos, est_pos, waypoints, current_wp_idx, spatial_memory_nodes, tree_positions, current_yaw, phase_str)

    cv2.imshow("Drone Sensor Dashboard & Telemetry", full_hud)
    cv2.imshow("Mario Kart Style Tactical Minimap", minimap_img)

    # Pop up final accuracy report window once mission completes
    if mission_complete and not report_displayed:
        report_displayed = True
        final_report_img = generate_final_accuracy_report(accuracy_history)
        cv2.imshow("Navigation Accuracy Report", final_report_img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
p.disconnect()