import cv2
import math
import random
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# Config
# =========================
INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output_firethrow.mp4"
MODEL_PATH = "hand_landmarker.task"

MAX_NUM_HANDS = 1
FIREBALL_RADIUS = 36

SMALL_BLUR = 31
LARGE_BLUR = 81

# Fire colors in BGR
CORE_COLOR = (220, 255, 255)      # white-yellow core
INNER_FIRE = (0, 220, 255)        # yellow
MID_FIRE = (0, 140, 255)          # orange
OUTER_FIRE = (0, 60, 255)         # red-orange
DARK_SMOKE = (40, 40, 80)

# Throw detection
THROW_SPEED_THRESHOLD = 22.0
THROW_DECAY = 0.88
BODY_BURN_RADIUS = 180

# MediaPipe landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


# =========================
# Utility
# =========================
def clamp_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def to_px(lm, width, height):
    return np.array([lm.x * width, lm.y * height], dtype=np.float32)


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def normalize(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n < 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)
    return v / n


def smooth_point(prev, curr, alpha=0.25):
    if prev is None:
        return curr
    return (1.0 - alpha) * prev + alpha * curr


def smooth_value(prev, curr, alpha=0.2):
    if prev is None:
        return curr
    return (1.0 - alpha) * prev + alpha * curr


# =========================
# Hand geometry
# =========================
def get_landmark_points(hand_landmarks, width, height):
    return [to_px(lm, width, height) for lm in hand_landmarks]


def compute_palm_center(pts):
    palm_ids = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    palm_pts = np.array([pts[i] for i in palm_ids], dtype=np.float32)
    center = palm_pts.mean(axis=0)
    mcp_mid = 0.5 * (pts[INDEX_MCP] + pts[PINKY_MCP])
    center = 0.65 * center + 0.35 * mcp_mid
    return center


def estimate_palm_facing_score(pts, handedness_name):
    wrist = pts[WRIST]
    middle = pts[MIDDLE_MCP]
    index_mcp = pts[INDEX_MCP]
    pinky_mcp = pts[PINKY_MCP]
    thumb_tip = pts[THUMB_TIP]

    a = middle - wrist
    b = pinky_mcp - index_mcp

    a_n = normalize(a)
    b_n = normalize(b)

    cross_z = a_n[0] * b_n[1] - a_n[1] * b_n[0]

    hand_width = max(norm(b), 1.0)
    thumb_side = (thumb_tip[0] - pts[PINKY_MCP][0]) / hand_width

    if handedness_name.lower() == "right":
        side_term = -thumb_side
    else:
        side_term = thumb_side

    score = 6.0 * cross_z + 2.0 * side_term

    # flipped so PALM visible => higher score
    return float(-score)


def compute_palm_side_offset(pts, handedness_name):
    wrist = pts[WRIST]
    middle = pts[MIDDLE_MCP]
    index_mcp = pts[INDEX_MCP]
    pinky_mcp = pts[PINKY_MCP]

    across = pinky_mcp - index_mcp
    across_n = normalize(across)
    perp = np.array([-across_n[1], across_n[0]], dtype=np.float32)

    wrist_to_middle = normalize(middle - wrist)
    if np.dot(perp, wrist_to_middle) < 0:
        perp = -perp

    base_offset = 18.0 * perp
    thumb_bias = 0.12 * across
    if handedness_name.lower() == "right":
        thumb_bias = -thumb_bias

    return base_offset + thumb_bias


def build_hand_occlusion_mask(frame_shape, pts):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    def poly(ids):
        arr = np.array([[int(pts[i][0]), int(pts[i][1])] for i in ids], dtype=np.int32)
        cv2.fillConvexPoly(mask, arr, 255)

    poly([WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP])
    poly([WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP])
    poly([INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP])
    poly([MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP])
    poly([RING_MCP, RING_PIP, RING_DIP, RING_TIP])
    poly([PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP])

    base_bridge = np.array([
        [int(pts[INDEX_MCP][0]), int(pts[INDEX_MCP][1])],
        [int(pts[MIDDLE_MCP][0]), int(pts[MIDDLE_MCP][1])],
        [int(pts[RING_MCP][0]), int(pts[RING_MCP][1])],
        [int(pts[PINKY_MCP][0]), int(pts[PINKY_MCP][1])],
    ], dtype=np.int32)
    cv2.fillConvexPoly(mask, base_bridge, 255)

    kernel = np.ones((9, 9), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


# =========================
# Fire rendering
# =========================
def add_radial_fire(glow_layer, core_layer, center, radius, intensity, frame_idx):
    cx, cy = int(center[0]), int(center[1])
    pulse = 1.0 + 0.10 * math.sin(frame_idx * 0.35)
    r = max(6, int(radius * pulse * (0.75 + 0.5 * intensity)))

    cv2.circle(glow_layer, (cx, cy), r + 30, OUTER_FIRE, -1, cv2.LINE_AA)
    cv2.circle(glow_layer, (cx, cy), r + 18, MID_FIRE, -1, cv2.LINE_AA)
    cv2.circle(glow_layer, (cx, cy), r + 8, INNER_FIRE, -1, cv2.LINE_AA)
    cv2.circle(core_layer, (cx, cy), r, CORE_COLOR, -1, cv2.LINE_AA)

    # Sparks
    spark_count = 8 + int(10 * intensity)
    for i in range(spark_count):
        ang = frame_idx * 0.09 + i * (2.0 * math.pi / max(1, spark_count))
        dist = r + 6 + random.uniform(0, 10 + 25 * intensity)
        px = int(cx + math.cos(ang) * dist)
        py = int(cy + math.sin(ang) * dist)
        sr = random.randint(2, 4)
        cv2.circle(glow_layer, (px, py), sr, INNER_FIRE, -1, cv2.LINE_AA)


def add_fire_streaks(glow_layer, center, velocity, intensity, width, height):
    speed = norm(velocity)
    if speed < 1.0:
        return

    dirn = normalize(velocity)
    trail_dir = -dirn
    base = np.array(center, dtype=np.float32)

    streaks = 5 + int(8 * intensity)
    for _ in range(streaks):
        side = np.array([-trail_dir[1], trail_dir[0]], dtype=np.float32)
        jitter_side = random.uniform(-20, 20)
        start = base + side * jitter_side
        length = random.uniform(30, 110 + 120 * intensity)

        mid = start + trail_dir * (length * 0.45) + side * random.uniform(-10, 10)
        end = start + trail_dir * length + side * random.uniform(-18, 18)

        pts = np.array([
            clamp_point(int(start[0]), int(start[1]), width, height),
            clamp_point(int(mid[0]), int(mid[1]), width, height),
            clamp_point(int(end[0]), int(end[1]), width, height),
        ], dtype=np.int32)

        cv2.polylines(glow_layer, [pts], False, MID_FIRE, thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(glow_layer, [pts], False, OUTER_FIRE, thickness=4, lineType=cv2.LINE_AA)


def create_fireball_layer(frame_shape, center_pt, radius, frame_idx, palm_visibility, velocity, throw_energy):
    h, w = frame_shape[:2]
    glow_layer = np.zeros((h, w, 3), dtype=np.uint8)
    core_layer = np.zeros((h, w, 3), dtype=np.uint8)

    visible_scale = 0.08 + 0.92 * palm_visibility
    fire_intensity = min(1.0, 0.35 + 0.65 * visible_scale + 0.75 * throw_energy)

    add_radial_fire(glow_layer, core_layer, center_pt, radius, fire_intensity, frame_idx)
    add_fire_streaks(glow_layer, center_pt, velocity, throw_energy, w, h)

    glow_small = cv2.GaussianBlur(glow_layer, (SMALL_BLUR, SMALL_BLUR), 0)
    glow_large = cv2.GaussianBlur(glow_layer, (LARGE_BLUR, LARGE_BLUR), 0)
    glow_combined = cv2.addWeighted(glow_small, 0.85, glow_large, 0.95, 0)

    glow_combined = np.clip(glow_combined.astype(np.float32) * visible_scale, 0, 255).astype(np.uint8)
    core_layer = np.clip(core_layer.astype(np.float32) * visible_scale, 0, 255).astype(np.uint8)

    return glow_combined, core_layer


def create_body_burn_layer(frame_shape, body_center, throw_energy, frame_idx):
    h, w = frame_shape[:2]
    burn = np.zeros((h, w, 3), dtype=np.uint8)

    if throw_energy < 0.05:
        return burn

    cx, cy = int(body_center[0]), int(body_center[1])
    aura_r = int(BODY_BURN_RADIUS * (0.75 + 0.6 * throw_energy))

    cv2.circle(burn, (cx, cy), aura_r, OUTER_FIRE, -1, cv2.LINE_AA)
    cv2.circle(burn, (cx, cy), int(aura_r * 0.7), MID_FIRE, -1, cv2.LINE_AA)

    # vertical flames rising around body
    flame_count = 20 + int(25 * throw_energy)
    for i in range(flame_count):
        base_x = int(cx + random.uniform(-aura_r * 0.7, aura_r * 0.7))
        base_y = int(cy + random.uniform(-aura_r * 0.2, aura_r * 0.9))

        flame_h = random.uniform(25, 70 + 90 * throw_energy)
        sway = random.uniform(-20, 20)

        p1 = (base_x, base_y)
        p2 = (int(base_x + sway * 0.4), int(base_y - flame_h * 0.45))
        p3 = (int(base_x + sway), int(base_y - flame_h))

        pts = np.array([p1, p2, p3], dtype=np.int32)
        cv2.polylines(burn, [pts], False, OUTER_FIRE, thickness=5, lineType=cv2.LINE_AA)
        cv2.polylines(burn, [pts], False, MID_FIRE, thickness=3, lineType=cv2.LINE_AA)

    burn_small = cv2.GaussianBlur(burn, (41, 41), 0)
    burn_large = cv2.GaussianBlur(burn, (91, 91), 0)
    burn = cv2.addWeighted(burn_small, 0.9, burn_large, 0.7, 0)

    burn = np.clip(burn.astype(np.float32) * min(1.0, 0.4 + 1.2 * throw_energy), 0, 255).astype(np.uint8)
    return burn


def composite_with_occlusion(frame, glow_layer, core_layer, hand_mask, palm_visibility, body_burn):
    out = cv2.add(frame, body_burn)
    out = cv2.add(out, glow_layer)
    out = cv2.add(out, core_layer)

    # Back of hand blocks fireball
    occlusion_strength = 1.0 - palm_visibility
    if occlusion_strength > 0.01:
        mask3 = cv2.merge([hand_mask, hand_mask, hand_mask]).astype(np.float32) / 255.0
        suppress = 0.94 * occlusion_strength
        out = out.astype(np.float32)
        base = (cv2.add(frame, body_burn)).astype(np.float32)
        out = out * (1.0 - suppress * mask3) + base * (suppress * mask3)
        out = np.clip(out, 0, 255).astype(np.uint8)

    return out


# =========================
# MediaPipe helpers
# =========================
def extract_hand(result):
    if not result.hand_landmarks:
        return None, None

    hand_landmarks = result.hand_landmarks[0]
    handedness_name = "Right"

    if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
        handedness_name = result.handedness[0][0].category_name

    return hand_landmarks, handedness_name


# =========================
# Main
# =========================
def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video: {OUTPUT_VIDEO}")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    smoothed_center = None
    smoothed_visibility = None
    prev_center = None
    throw_energy = 0.0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_landmarks, handedness_name = extract_hand(result)

            if hand_landmarks is not None:
                pts = get_landmark_points(hand_landmarks, width, height)

                palm_center = compute_palm_center(pts)
                facing_score = estimate_palm_facing_score(pts, handedness_name)

                x = float(np.clip(facing_score, -20.0, 20.0))
                palm_visible = 1.0 / (1.0 + math.exp(-x))
                palm_visible = float(np.clip(palm_visible, 0.0, 1.0))

                offset = compute_palm_side_offset(pts, handedness_name)
                orb_center = palm_center + offset
                orb_center[0], orb_center[1] = clamp_point(int(orb_center[0]), int(orb_center[1]), width, height)

                smoothed_center = smooth_point(
                    smoothed_center,
                    np.array(orb_center, dtype=np.float32),
                    alpha=0.25,
                )

                smoothed_visibility = smooth_value(smoothed_visibility, palm_visible, alpha=0.18)

                if prev_center is None:
                    velocity = np.array([0.0, 0.0], dtype=np.float32)
                else:
                    velocity = smoothed_center - prev_center

                speed = norm(velocity)
                prev_center = smoothed_center.copy()

                # Throw trigger
                if speed > THROW_SPEED_THRESHOLD and smoothed_visibility > 0.45:
                    throw_energy = min(1.0, throw_energy + 0.35)
                else:
                    throw_energy *= THROW_DECAY

                hand_mask = build_hand_occlusion_mask(frame.shape, pts)

                glow_layer, core_layer = create_fireball_layer(
                    frame.shape,
                    smoothed_center,
                    FIREBALL_RADIUS,
                    frame_idx,
                    smoothed_visibility,
                    velocity,
                    throw_energy,
                )

                # approximate body center lower than frame center
                body_center = np.array([width * 0.5, height * 0.62], dtype=np.float32)
                body_burn = create_body_burn_layer(frame.shape, body_center, throw_energy, frame_idx)

                output_frame = composite_with_occlusion(
                    frame,
                    glow_layer,
                    core_layer,
                    hand_mask,
                    smoothed_visibility,
                    body_burn,
                )
            else:
                throw_energy *= THROW_DECAY
                body_center = np.array([width * 0.5, height * 0.62], dtype=np.float32)
                body_burn = create_body_burn_layer(frame.shape, body_center, throw_energy, frame_idx)
                output_frame = cv2.add(frame, body_burn)

            writer.write(output_frame)
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"Done. Saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()