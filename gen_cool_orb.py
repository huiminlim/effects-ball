import cv2
import math
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# Config
# =========================
INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output_lightball.mp4"
MODEL_PATH = "hand_landmarker.task"

MAX_NUM_HANDS = 1
BALL_RADIUS = 34
GLOW_BLUR_SMALL = 31
GLOW_BLUR_LARGE = 71

# BGR colors
CORE_COLOR = (255, 255, 255)
INNER_GLOW_COLOR = (255, 180, 80)
OUTER_GLOW_COLOR = (255, 90, 20)

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


# =========================
# Hand geometry
# =========================
def get_landmark_points(hand_landmarks, width, height):
    pts = []
    for lm in hand_landmarks:
        pts.append(to_px(lm, width, height))
    return pts


def compute_palm_center(pts):
    palm_ids = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    palm_pts = np.array([pts[i] for i in palm_ids], dtype=np.float32)
    center = palm_pts.mean(axis=0)

    mcp_mid = 0.5 * (pts[INDEX_MCP] + pts[PINKY_MCP])
    center = 0.65 * center + 0.35 * mcp_mid
    return center


def estimate_palm_facing_score(pts, handedness_name):
    """
    Stable estimate:
    positive score should mean PALM facing camera
    negative score should mean BACK OF HAND facing camera
    """
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

    # IMPORTANT:
    # The previous version felt reversed.
    # Flip the sign so palm-facing becomes visible and back-of-hand hides it.
    score = -score

    return float(score)


def compute_palm_side_offset(pts, handedness_name, palm_visible):
    """
    Compute a small 2D offset so the orb sits more toward the inner palm side.

    When palm is visible, push the orb toward the perceived palm interior.
    When back of hand is visible, keep the same hidden-side location so the hand blocks it.
    """
    wrist = pts[WRIST]
    middle = pts[MIDDLE_MCP]
    index_mcp = pts[INDEX_MCP]
    pinky_mcp = pts[PINKY_MCP]

    across = pinky_mcp - index_mcp
    across_n = normalize(across)

    # Perpendicular to hand width
    perp = np.array([-across_n[1], across_n[0]], dtype=np.float32)

    # Choose palm direction based on wrist->middle direction
    wrist_to_middle = normalize(middle - wrist)

    if np.dot(perp, wrist_to_middle) < 0:
        perp = -perp

    # Small inward palm shift
    base_offset = 18.0 * perp

    # Slight thumb-side bias to make it feel like "holding" in the palm
    thumb_bias = 0.12 * across
    if handedness_name.lower() == "right":
        thumb_bias = -thumb_bias

    # Keep the orb on the hidden inner-hand side even when the back is visible.
    # So do NOT reverse it based on visibility.
    return base_offset + thumb_bias


def build_hand_occlusion_mask(frame_shape, pts):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    def poly(ids):
        arr = np.array(
            [[int(pts[i][0]), int(pts[i][1])] for i in ids],
            dtype=np.int32
        )
        cv2.fillConvexPoly(mask, arr, 255)

    # Palm
    poly([WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP])

    # Thumb
    poly([WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP])

    # Fingers
    poly([INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP])
    poly([MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP])
    poly([RING_MCP, RING_PIP, RING_DIP, RING_TIP])
    poly([PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP])

    # Base bridge
    base_bridge = np.array([
        [int(pts[INDEX_MCP][0]), int(pts[INDEX_MCP][1])],
        [int(pts[MIDDLE_MCP][0]), int(pts[MIDDLE_MCP][1])],
        [int(pts[RING_MCP][0]), int(pts[RING_MCP][1])],
        [int(pts[PINKY_MCP][0]), int(pts[PINKY_MCP][1])],
    ], dtype=np.int32)
    cv2.fillConvexPoly(mask, base_bridge, 255)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# =========================
# Rendering
# =========================
def create_light_ball_layer(frame_shape, center_pt, radius, frame_idx, palm_visibility):
    """
    palm_visibility:
      1.0 => palm visible, strong glow
      0.0 => back of hand, glow mostly hidden
    """
    h, w = frame_shape[:2]
    glow_layer = np.zeros((h, w, 3), dtype=np.uint8)
    core_layer = np.zeros((h, w, 3), dtype=np.uint8)

    cx, cy = int(center_pt[0]), int(center_pt[1])

    pulse = 1.0 + 0.08 * math.sin(frame_idx * 0.35)
    r = max(8, int(radius * pulse))

    visible_scale = 0.10 + 0.90 * palm_visibility
    rr = max(4, int(r * (0.6 + 0.4 * visible_scale)))

    cv2.circle(glow_layer, (cx, cy), rr + 30, OUTER_GLOW_COLOR, -1, cv2.LINE_AA)
    cv2.circle(glow_layer, (cx, cy), rr + 16, INNER_GLOW_COLOR, -1, cv2.LINE_AA)
    cv2.circle(core_layer, (cx, cy), rr, CORE_COLOR, -1, cv2.LINE_AA)

    for i in range(6):
        ang = frame_idx * 0.08 + i * (2.0 * math.pi / 6.0)
        dist = rr + 8 + 4 * math.sin(frame_idx * 0.2 + i)
        px = int(cx + math.cos(ang) * dist)
        py = int(cy + math.sin(ang) * dist)
        cv2.circle(glow_layer, (px, py), 3, INNER_GLOW_COLOR, -1, cv2.LINE_AA)

    glow_small = cv2.GaussianBlur(glow_layer, (GLOW_BLUR_SMALL, GLOW_BLUR_SMALL), 0)
    glow_large = cv2.GaussianBlur(glow_layer, (GLOW_BLUR_LARGE, GLOW_BLUR_LARGE), 0)
    glow_combined = cv2.addWeighted(glow_small, 0.8, glow_large, 0.9, 0)

    glow_combined = np.clip(glow_combined.astype(np.float32) * visible_scale, 0, 255).astype(np.uint8)
    core_layer = np.clip(core_layer.astype(np.float32) * visible_scale, 0, 255).astype(np.uint8)

    return glow_combined, core_layer


def composite_with_occlusion(frame, glow_layer, core_layer, hand_mask, palm_visibility):
    """
    Always place the orb at the palm-side location.
    When back of hand faces camera, hand should block it more strongly.
    """
    out = cv2.add(frame, glow_layer)
    out = cv2.add(out, core_layer)

    occlusion_strength = 1.0 - palm_visibility
    if occlusion_strength > 0.01:
        mask3 = cv2.merge([hand_mask, hand_mask, hand_mask]).astype(np.float32) / 255.0
        suppress = 0.92 * occlusion_strength
        out = out.astype(np.float32)
        out = out * (1.0 - suppress * mask3) + frame.astype(np.float32) * (suppress * mask3)
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

                # Safe sigmoid
                x = float(np.clip(facing_score, -20.0, 20.0))
                palm_visible = 1.0 / (1.0 + math.exp(-x))
                palm_visible = float(np.clip(palm_visible, 0.0, 1.0))

                offset = compute_palm_side_offset(pts, handedness_name, palm_visible)
                orb_center = palm_center + offset

                orb_center[0], orb_center[1] = clamp_point(
                    int(orb_center[0]), int(orb_center[1]), width, height
                )

                smoothed_center = smooth_point(
                    smoothed_center,
                    np.array(orb_center, dtype=np.float32),
                    alpha=0.25,
                )

                if smoothed_visibility is None:
                    smoothed_visibility = palm_visible
                else:
                    smoothed_visibility = 0.82 * smoothed_visibility + 0.18 * palm_visible

                hand_mask = build_hand_occlusion_mask(frame.shape, pts)

                glow_layer, core_layer = create_light_ball_layer(
                    frame.shape,
                    smoothed_center,
                    BALL_RADIUS,
                    frame_idx,
                    smoothed_visibility,
                )

                output_frame = composite_with_occlusion(
                    frame,
                    glow_layer,
                    core_layer,
                    hand_mask,
                    smoothed_visibility,
                )
            else:
                output_frame = frame

            writer.write(output_frame)
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"Done. Saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()