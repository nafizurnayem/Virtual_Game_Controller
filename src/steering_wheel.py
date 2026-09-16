import cv2
import mediapipe as mp
import numpy as np
import math
import time
import platform
import threading
from collections import deque
from pynput.keyboard import Key, Controller

CAMERA_INDEX       = 1
CAMERA_WIDTH       = 640
CAMERA_HEIGHT      = 480
CAMERA_FPS         = 60      # request the highest the camera will give; frame interval is the floor on input lag
DEAD_ZONE_DEG      = 12
RELEASE_ZONE_DEG   = 6
SOFT_ZONE_DEG      = 25
FLIP_CAMERA        = False
SHOW_ANGLE         = True
WINDOW_NAME        = "Virtual Steering Wheel"
ALWAYS_ON_TOP      = True      # keep the preview above your game window
WINDOW_POS         = (20, 20)  # starting position; drag the window anywhere after launch
WINDOW_SIZE        = (960, 720) # starting preview size; user can resize the window
MIN_DETECTION_CONF = 0.7
MIN_TRACKING_CONF  = 0.5
GRACE_FRAMES       = 8
OPEN_FINGER_THRESH = 3

GAME_MODE          = "RACING"  # "RACING" (hold keys: wheel + fist/open throttle) or "RUNNER" (one-hand 2-finger swipe)
RUNNER_INVERT_X    = False     # set True if left/right swipes come out reversed for your camera

# Swipe detection. Distances are in "frame heights" (x is aspect-corrected), so
# they mean the same thing regardless of resolution; timings are in seconds, so
# they mean the same thing regardless of FPS.
SWIPE_TIME_WINDOW  = 0.18      # seconds of fingertip history examined for a swipe
SWIPE_MIN_SAMPLES  = 3         # frames needed before a swipe can fire (lower = snappier, noisier)
SWIPE_DIST_THRESH  = 0.075     # travel needed to fire (lower = fires earlier in the motion)
SWIPE_MIN_SPEED    = 0.55      # frame-heights/sec; rejects slow hand drift
SWIPE_AXIS_RATIO   = 1.4       # dominant axis must beat the other by this much, else ignore as diagonal
SWIPE_COOLDOWN     = 0.13      # min seconds between swipes
SWIPE_SETTLE_SPEED = 0.45      # hand must slow below this to re-arm the next swipe
SMOOTHING_ALPHA    = 0.7       # fingertip EMA smoothing (1.0 = raw/jittery, lower = smoother/laggier)
POSE_GRACE_FRAMES  = 4         # pose may flicker off this many frames mid-swipe without resetting
SHOW_SWIPE_TRAIL   = True      # draw the tracked fingertip path in runner mode

# Paddle shifters (racing mode). Flick two fingers up on one hand to shift:
# right hand = up a gear, left hand = down a gear.
PADDLE_ENABLED       = True
GEAR_UP_KEY          = 'e'     # rebind to whatever your game uses for upshift
GEAR_DOWN_KEY        = 'q'     # ...and downshift
PADDLE_SWAP_HANDS    = False   # True if the paddles come out backwards for your camera
PADDLE_HOLD_FRAMES   = 3       # frames the flick must persist; guards against fist->open transitions
PADDLE_RELEASE_FRAMES = 2      # frames the flick must be gone before the next shift arms
PADDLE_COOLDOWN      = 0.22    # min seconds between shifts
GEAR_MIN             = 1
GEAR_MAX             = 8
SHIFT_FLASH_SECONDS  = 0.6     # how long the shift indicator stays lit on the HUD

CLR_WHEEL   = (80, 200, 255)
CLR_LEFT    = (60, 120, 255)
CLR_RIGHT   = (50, 220, 140)
CLR_NEUTRAL = (200, 200, 200)
CLR_TEXT    = (255, 255, 255)
CLR_ACCENT  = (0, 180, 255)
CLR_HAND_L  = (255, 130, 60)
CLR_HAND_R  = (60, 230, 130)
CLR_ACCEL   = (50, 220, 100)
CLR_BRAKE   = (0, 60, 255)

keyboard   = Controller()
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def is_open_hand(hand_landmarks):
    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]
    extended = sum(
        1 for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
    )
    return extended >= OPEN_FINGER_THRESH


EXTEND_RATIO = 1.15


def _is_finger_extended(lm, tip_id, pip_id):
    """Extended = fingertip sits farther from the wrist than its middle joint.

    Distance-based rather than "tip.y < pip.y" so the test survives the hand
    tilting or rotating, which it always does during a swipe.
    """
    wrist = lm[0]
    d_tip = math.hypot(lm[tip_id].x - wrist.x, lm[tip_id].y - wrist.y)
    d_pip = math.hypot(lm[pip_id].x - wrist.x, lm[pip_id].y - wrist.y)
    return d_tip > d_pip * EXTEND_RATIO


def is_two_finger_pose(hand_landmarks):
    """Index + middle extended, ring + pinky curled — the runner-mode 'trigger' pose."""
    lm = hand_landmarks.landmark
    return (
        _is_finger_extended(lm, 8, 6)
        and _is_finger_extended(lm, 12, 10)
        and not _is_finger_extended(lm, 16, 14)
        and not _is_finger_extended(lm, 20, 18)
    )


class SteeringController:
    def __init__(self):
        self.keys_held     = {Key.left: False, Key.right: False, Key.up: False, Key.down: False}
        self.angle_history = []
        self.HISTORY_LEN   = 1

    def _press(self, key):
        if not self.keys_held[key]:
            keyboard.press(key)
            self.keys_held[key] = True

    def _release(self, key):
        if self.keys_held[key]:
            keyboard.release(key)
            self.keys_held[key] = False

    def release_all(self):
        for key in list(self.keys_held.keys()):
            try:
                keyboard.release(key)
            except Exception:
                pass
            self.keys_held[key] = False
        self.angle_history.clear()

    def smooth_angle(self, raw_angle):
        self.angle_history.append(raw_angle)
        if len(self.angle_history) > self.HISTORY_LEN:
            self.angle_history.pop(0)
        return float(np.mean(self.angle_history))

    def update_steer(self, left_wrist, right_wrist):
        dx = right_wrist[0] - left_wrist[0]
        dy = right_wrist[1] - left_wrist[1]

        raw_angle_rad = math.atan2(dy, dx)
        raw_angle_deg = math.degrees(raw_angle_rad)
        angle = self.smooth_angle(raw_angle_deg)

        direction = "STRAIGHT"
        if angle < -DEAD_ZONE_DEG:
            direction = "LEFT"
        elif angle > DEAD_ZONE_DEG:
            direction = "RIGHT"
        elif self.keys_held[Key.left] and angle > -RELEASE_ZONE_DEG:
            direction = "STRAIGHT"
        elif self.keys_held[Key.right] and angle < RELEASE_ZONE_DEG:
            direction = "STRAIGHT"

        strength = 0.0
        if direction == "LEFT":
            strength = min(1.0, (abs(angle) - DEAD_ZONE_DEG) / (SOFT_ZONE_DEG - DEAD_ZONE_DEG))
            self._press(Key.left)
            self._release(Key.right)
        elif direction == "RIGHT":
            strength = min(1.0, (abs(angle) - DEAD_ZONE_DEG) / (SOFT_ZONE_DEG - DEAD_ZONE_DEG))
            self._press(Key.right)
            self._release(Key.left)
        else:
            self._release(Key.left)
            self._release(Key.right)

        return angle, direction, strength

    def update_throttle(self, left_open, right_open):
        both_open  = left_open and right_open
        both_fist  = (not left_open) and (not right_open)

        if both_fist:
            self._press(Key.up)
            self._release(Key.down)
            return "ACCEL"
        elif both_open:
            self._press(Key.down)
            self._release(Key.up)
            return "BRAKE"
        else:
            self._release(Key.up)
            self._release(Key.down)
            return "NEUTRAL"


class PaddleShifter:
    """Two-finger flick on one hand acts as a paddle shifter.

    Right hand flick = up a gear, left hand = down a gear. Each hand is
    edge-triggered: the flick must hold for a few frames to fire, then fully
    clear before that paddle arms again, so holding two fingers up does not
    rattle through the gearbox.

    The hold requirement matters because going from a fist to an open hand
    passes through a moment where the index and middle fingers are extended
    while the ring and pinky are still curled — that is the paddle pose. A few
    frames of persistence tells a deliberate flick apart from that transition.
    """

    HANDS = ("Left", "Right")

    def __init__(self):
        self.hold_frames  = {h: 0 for h in self.HANDS}
        self.clear_frames = {h: 0 for h in self.HANDS}
        self.armed        = {h: True for h in self.HANDS}
        # Per paddle, so an upshift never rate-limits a downshift.
        self.last_shift_time = {h: 0.0 for h in self.HANDS}
        self.gear            = GEAR_MIN
        self.last_shift      = None
        self.last_shift_at   = 0.0

    def reset(self):
        for h in self.HANDS:
            self.hold_frames[h]      = 0
            self.clear_frames[h]     = 0
            self.armed[h]            = True
            self.last_shift_time[h]  = 0.0
        self.gear          = GEAR_MIN
        self.last_shift    = None
        self.last_shift_at = 0.0

    def _shift(self, hand):
        """Fire the paddle. Returns False if the cooldown suppressed it, in which
        case the caller leaves the paddle armed so the flick still lands once the
        cooldown expires rather than being silently dropped."""
        now = time.time()
        if now - self.last_shift_time[hand] < PADDLE_COOLDOWN:
            return False

        up = (hand == "Right") != PADDLE_SWAP_HANDS
        key = GEAR_UP_KEY if up else GEAR_DOWN_KEY

        keyboard.press(key)
        keyboard.release(key)

        self.gear = min(GEAR_MAX, self.gear + 1) if up else max(GEAR_MIN, self.gear - 1)
        self.last_shift           = "UP" if up else "DOWN"
        self.last_shift_at        = now
        self.last_shift_time[hand] = now
        return True

    def update(self, flicks):
        """flicks: {"Left": bool, "Right": bool} — is the paddle pose showing?"""
        if not PADDLE_ENABLED:
            return

        for hand in self.HANDS:
            if flicks.get(hand, False):
                self.hold_frames[hand] += 1
                self.clear_frames[hand] = 0
                if self.armed[hand] and self.hold_frames[hand] >= PADDLE_HOLD_FRAMES:
                    if self._shift(hand):
                        self.armed[hand] = False
            else:
                self.clear_frames[hand] += 1
                if self.clear_frames[hand] >= PADDLE_RELEASE_FRAMES:
                    self.hold_frames[hand] = 0
                    self.armed[hand]       = True

    def flash(self):
        """The recent shift to show on the HUD, or None once it has faded."""
        if self.last_shift is None:
            return None
        if time.time() - self.last_shift_at > SHIFT_FLASH_SECONDS:
            return None
        return self.last_shift


class RunnerController:
    """One-hand swipe controls for lane-runner games (Temple Run style).

    Hold up a 'two-finger' pose (index + middle extended, ring + pinky curled)
    and swipe that hand: up=jump, down=slide, left/right=lane change.

    A swipe fires when the smoothed fingertip travels far enough, fast enough,
    and clearly enough along one axis inside a short time window. After firing,
    the hand must slow to a near-stop before the next swipe arms — that is what
    keeps the return stroke of a gesture from firing the opposite direction.
    """

    def __init__(self):
        self.history         = deque()
        self.smoothed        = None
        self.missing_frames  = 0
        self.last_swipe_time = 0.0
        self.needs_settle    = True
        self.last_action     = "READY"

    def _reset_motion(self):
        self.history.clear()
        self.smoothed = None
        # Acquiring the pose means the hand was just moving into frame; require
        # it to come to rest before the first swipe can fire.
        self.needs_settle = True

    def release_all(self):
        self._reset_motion()
        self.missing_frames = 0
        self.last_action    = "READY"

    def trail(self):
        return [(x, y) for _, x, y in self.history]

    def _track(self, point, aspect, now):
        """Aspect-correct, EMA-smooth, and record the fingertip position."""
        x = point[0] * aspect
        y = point[1]

        if self.smoothed is None:
            self.smoothed = (x, y)
        else:
            sx, sy = self.smoothed
            self.smoothed = (
                SMOOTHING_ALPHA * x + (1.0 - SMOOTHING_ALPHA) * sx,
                SMOOTHING_ALPHA * y + (1.0 - SMOOTHING_ALPHA) * sy,
            )

        self.history.append((now, self.smoothed[0], self.smoothed[1]))
        while len(self.history) > 1 and now - self.history[0][0] > SWIPE_TIME_WINDOW:
            self.history.popleft()

    def _recent_speed(self):
        """Speed over just the last few samples — how fast the hand is moving now."""
        if len(self.history) < 2:
            return 0.0
        i = max(0, len(self.history) - SWIPE_MIN_SAMPLES)
        t0, x0, y0 = self.history[i]
        t1, x1, y1 = self.history[-1]
        dt = t1 - t0
        if dt <= 0.0:
            return 0.0
        return math.hypot(x1 - x0, y1 - y0) / dt

    def _find_swipe(self):
        """Scan every sub-window ending at the newest sample, return the largest
        travel that clears both gates.

        Measuring across the whole window would divide the distance by the full
        window duration, so the stationary moments before a flick drag the
        computed speed down and the gesture has to travel further before it
        qualifies. Checking sub-windows lets a fast flick fire the moment it is
        genuinely fast, while slow drift still fails at every span.
        """
        t_end, x_end, y_end = self.history[-1]
        best = None

        for i in range(len(self.history) - SWIPE_MIN_SAMPLES + 1):
            t0, x0, y0 = self.history[i]
            dt = t_end - t0
            if dt <= 0.0:
                continue
            dx, dy = x_end - x0, y_end - y0
            dist = math.hypot(dx, dy)
            if dist < SWIPE_DIST_THRESH or dist / dt < SWIPE_MIN_SPEED:
                continue
            if best is None or dist > best[2]:
                best = (dx, dy, dist)

        return best

    def update(self, pose_active, point, aspect):
        now = time.time()

        if not pose_active:
            self.missing_frames += 1
            if self.missing_frames > POSE_GRACE_FRAMES:
                self._reset_motion()
                self.last_action = "READY"
            return self.last_action

        self.missing_frames = 0
        self._track(point, aspect, now)

        # Flush the buffer during cooldown so the hand pulling back out of the
        # swipe never accumulates into a bogus opposite-direction gesture.
        if now - self.last_swipe_time < SWIPE_COOLDOWN:
            self.history.clear()
            return self.last_action

        if len(self.history) < SWIPE_MIN_SAMPLES:
            return self.last_action

        if self.needs_settle:
            if self._recent_speed() < SWIPE_SETTLE_SPEED:
                self.needs_settle = False
                self.last_action  = "READY"
            return self.last_action

        swipe = self._find_swipe()
        if swipe is None:
            self.last_action = "READY"
            return self.last_action
        dx, dy, _ = swipe

        if abs(dx) > abs(dy) * SWIPE_AXIS_RATIO:
            if RUNNER_INVERT_X:
                dx = -dx
            action, key = ("RIGHT", Key.right) if dx > 0 else ("LEFT", Key.left)
        elif abs(dy) > abs(dx) * SWIPE_AXIS_RATIO:
            action, key = ("SLIDE", Key.down) if dy > 0 else ("JUMP", Key.up)
        else:
            self.last_action = "READY"   # ambiguous diagonal — don't guess
            return self.last_action

        keyboard.press(key)
        keyboard.release(key)
        self.last_swipe_time = now
        self.needs_settle    = True
        self.last_action     = action
        self.history.clear()
        return action


def draw_steering_wheel(frame, center, angle_deg, direction, strength):
    h, w = frame.shape[:2]
    radius = int(min(w, h) * 0.10)
    cx, cy = center

    color = CLR_NEUTRAL
    if direction == "LEFT":
        color = CLR_LEFT
    elif direction == "RIGHT":
        color = CLR_RIGHT

    cv2.circle(frame, (cx + 3, cy + 3), radius, (0, 0, 0), 4)
    cv2.circle(frame, (cx, cy), radius, color, 3)

    for sa in [0, 120, 240]:
        rad = math.radians(sa - angle_deg)
        x1 = int(cx + radius * 0.4 * math.cos(rad))
        y1 = int(cy - radius * 0.4 * math.sin(rad))
        x2 = int(cx + radius * 0.95 * math.cos(rad))
        y2 = int(cy - radius * 0.95 * math.sin(rad))
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    cv2.circle(frame, (cx, cy), 6, color, -1)

    if direction != "STRAIGHT":
        start_a = -30 if direction == "RIGHT" else 150
        end_a   =  30 if direction == "RIGHT" else 210
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_a, end_a, color, 5)


def draw_gear_box(frame, gear, flash):
    """Gear readout with the paddle indicators either side of it."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cx, cy = 70, h - 150

    cv2.rectangle(frame, (cx - 42, cy - 34), (cx + 42, cy + 26), (25, 25, 35), -1)
    cv2.rectangle(frame, (cx - 42, cy - 34), (cx + 42, cy + 26), CLR_ACCENT, 2)

    label = str(gear)
    (tw, _), _ = cv2.getTextSize(label, font, 1.2, 3)
    cv2.putText(frame, label, (cx - tw // 2, cy + 14), font, 1.2, CLR_TEXT, 3)
    cv2.putText(frame, "GEAR", (cx - 22, cy - 40), font, 0.4, CLR_ACCENT, 1)

    down_color = CLR_BRAKE if flash == "DOWN" else (70, 70, 85)
    up_color   = CLR_ACCEL if flash == "UP"   else (70, 70, 85)
    cv2.putText(frame, "-", (cx - 66, cy + 10), font, 0.9, down_color, 3)
    cv2.putText(frame, "+", (cx + 50, cy + 10), font, 0.9, up_color,   3)


def draw_hud(frame, angle, direction, strength, throttle_mode, both_hands_visible, left_open, right_open, fps,
             gear=None, shift_flash=None):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 160), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    bar_w = int(w * 0.5)
    bar_h = 14
    bar_x = (w - bar_w) // 2
    bar_y = h - 110
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)

    mid = bar_x + bar_w // 2
    cv2.rectangle(frame, (mid - 2, bar_y - 4), (mid + 2, bar_y + bar_h + 4), (180, 180, 180), -1)

    fill_len = int((bar_w // 2) * strength)
    if direction == "LEFT" and fill_len > 0:
        cv2.rectangle(frame, (mid - fill_len, bar_y), (mid, bar_y + bar_h), CLR_LEFT, -1)
    elif direction == "RIGHT" and fill_len > 0:
        cv2.rectangle(frame, (mid, bar_y), (mid + fill_len, bar_y + bar_h), CLR_RIGHT, -1)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    dir_color = CLR_LEFT if direction == "LEFT" else (CLR_RIGHT if direction == "RIGHT" else CLR_NEUTRAL)
    cv2.putText(frame, "<- LEFT",  (bar_x, bar_y - 10),               font, 0.45, CLR_LEFT,  1)
    cv2.putText(frame, "RIGHT ->", (bar_x + bar_w - 80, bar_y - 10),  font, 0.45, CLR_RIGHT, 1)
    cv2.putText(frame, direction,  (mid - 30, bar_y + bar_h + 28),    font, 0.8,  dir_color, 2)

    if SHOW_ANGLE:
        cv2.putText(frame, f"{angle:+.1f} deg", (bar_x, h - 80), font, 0.55, CLR_TEXT, 1)

    throttle_color = CLR_ACCEL if throttle_mode == "ACCEL" else (CLR_BRAKE if throttle_mode == "BRAKE" else CLR_NEUTRAL)
    throttle_label = {
        "ACCEL":   "ACCEL [UP]",
        "BRAKE":   "BRAKE [DOWN]",
        "NEUTRAL": "NEUTRAL",
    }[throttle_mode]

    cv2.rectangle(frame, (bar_x, h - 65), (bar_x + bar_w, h - 42), (30, 30, 40), -1)
    cv2.rectangle(frame, (bar_x, h - 65), (bar_x + bar_w, h - 42), throttle_color, 2)
    cv2.putText(frame, throttle_label, (bar_x + 10, h - 48), font, 0.65, throttle_color, 2)

    l_label = "OPEN" if left_open else "FIST"
    r_label = "OPEN" if right_open else "FIST"
    l_color = CLR_BRAKE if left_open else CLR_ACCEL
    r_color = CLR_BRAKE if right_open else CLR_ACCEL
    cv2.putText(frame, f"L:{l_label}", (bar_x + bar_w + 10, h - 100), font, 0.5, l_color, 1)
    cv2.putText(frame, f"R:{r_label}", (bar_x + bar_w + 10, h - 80),  font, 0.5, r_color, 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 90, 30), font, 0.55, CLR_ACCENT, 1)

    status       = "BOTH HANDS DETECTED" if both_hands_visible else "SHOW BOTH HANDS"
    status_color = (60, 220, 60) if both_hands_visible else (0, 80, 255)
    cv2.putText(frame, status, (10, 30), font, 0.55, status_color, 1)

    draw_steering_wheel(frame, (w - 80, h - 80), angle, direction, strength)

    if gear is not None:
        draw_gear_box(frame, gear, shift_flash)


def draw_swipe_trail(frame, trail, aspect):
    """Draw the smoothed fingertip path so the gesture is visible as it happens."""
    h, w = frame.shape[:2]
    points = [(int((x / aspect) * w), int(y * h)) for x, y in trail]

    for i in range(1, len(points)):
        fade = i / len(points)
        thickness = max(1, int(1 + 4 * fade))
        cv2.line(frame, points[i - 1], points[i], CLR_ACCENT, thickness)


def draw_runner_hud(frame, action, pose_active, armed, fps):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 110), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = {
        "LEFT":  CLR_LEFT,
        "RIGHT": CLR_RIGHT,
        "JUMP":  CLR_ACCEL,
        "SLIDE": CLR_BRAKE,
        "READY": CLR_NEUTRAL,
    }
    color = colors.get(action, CLR_NEUTRAL)

    text = action
    (tw, th), _ = cv2.getTextSize(text, font, 1.4, 3)
    cv2.putText(frame, text, ((w - tw) // 2, h - 40), font, 1.4, color, 3)

    cv2.putText(frame, "RUNNER MODE  (index+middle up, swipe: up/down/left/right, M=switch mode, F=mirror)",
                (10, h - 85), font, 0.5, CLR_TEXT, 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 90, 30), font, 0.55, CLR_ACCENT, 1)

    if not pose_active:
        status, status_color = "SHOW INDEX + MIDDLE FINGER (ONE HAND)", (0, 80, 255)
    elif not armed:
        status, status_color = "HOLD STILL TO ARM", (0, 190, 255)
    else:
        status, status_color = "ARMED - SWIPE NOW", (60, 220, 60)
    cv2.putText(frame, status, (10, 30), font, 0.55, status_color, 1)


def draw_hand_connection(frame, lw, rw):
    lx, ly = lw
    rx, ry = rw
    cv2.line(frame, (lx, ly), (rx, ry), (30, 100, 200), 8)
    cv2.line(frame, (lx, ly), (rx, ry), CLR_ACCENT, 2)
    cv2.circle(frame, (lx, ly), 10, CLR_HAND_L, -1)
    cv2.circle(frame, (rx, ry), 10, CLR_HAND_R, -1)
    cv2.circle(frame, (lx, ly), 13, CLR_HAND_L, 2)
    cv2.circle(frame, (rx, ry), 13, CLR_HAND_R, 2)
    mx = (lx + rx) // 2
    my = (ly + ry) // 2
    cv2.circle(frame, (mx, my), 7, CLR_WHEEL, -1)


class FrameGrabber:
    """Pulls frames on a background thread and keeps only the newest one.

    cap.read() blocks for a full frame interval, so reading inline forces
    capture and inference to run back to back. Grabbing on its own thread lets
    them overlap, and dropping any frame the main loop was too busy to consume
    means the pose you see is always the most recent one the camera produced.
    """

    def __init__(self, cap):
        self.cap     = cap
        self.lock    = threading.Lock()
        self.frame   = None
        self.running = True
        self.thread  = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        """Return the newest unprocessed frame, or None if none has arrived."""
        with self.lock:
            frame, self.frame = self.frame, None
        return frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)


def keep_window_on_top():
    """Re-apply topmost state. Some Windows camera/backend events can reset it."""
    if not ALWAYS_ON_TOP:
        return
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except (cv2.error, AttributeError):
        pass


def setup_window():
    """Create a resizable, movable preview window and pin it above other windows.

    Note this cannot cover a game running in exclusive fullscreen — no
    always-on-top window can. Run the game borderless/windowed to see the HUD.
    """
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if WINDOW_SIZE is not None:
        cv2.resizeWindow(WINDOW_NAME, WINDOW_SIZE[0], WINDOW_SIZE[1])

    if WINDOW_POS is not None:
        cv2.moveWindow(WINDOW_NAME, WINDOW_POS[0], WINDOW_POS[1])

    keep_window_on_top()


def make_hands(game_mode):
    """Runner mode tracks a single hand, so it can afford the accurate landmark
    model (complexity 1) for roughly the cost of racing's two-hand fast model."""
    single = game_mode == "RUNNER"
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1 if single else 2,
        model_complexity=1 if single else 0,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    )


def _configure(cap):
    """Apply capture format. MUST run before the first read: once the stream is
    started, MSMF ignores format changes while still *reporting* the requested
    value back, so a late CAP_PROP_FPS silently leaves you at 30fps."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)


def measure_fps(cap, samples=20):
    """Time actual delivery; CAP_PROP_FPS reports what was asked for, not what arrives."""
    for _ in range(5):
        cap.read()
    start = time.perf_counter()
    got = 0
    for _ in range(samples):
        ok, _ = cap.read()
        if ok:
            got += 1
    elapsed = time.perf_counter() - start
    return got / elapsed if elapsed > 0 and got else 0.0


def open_camera(start_index=0):
    """Try start_index first, then scan nearby indices/backends.

    DroidCam's virtual webcam often fails to open (or opens but returns no
    frames) on Windows MSMF/CAP_ANY, and it rarely lands at index 0/1 since it
    registers after real webcams. Try DSHOW first on Windows, verify with an
    actual frame read (not just isOpened), and fall back across indices.
    """
    if platform.system() == "Darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    elif platform.system() == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    start_index = start_index % 4
    indices = [(start_index + i) % 4 for i in range(4)]

    for index in indices:
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            _configure(cap)
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"[INFO] Camera opened at index {index} (backend {backend})")
                return cap, index
            cap.release()

    return None, None


def switch_camera(cap, grabber, cam_index, controller, runner_controller):
    """Stop the current capture, open the next working camera, and return the new state.

    Returns (cap, grabber, cam_index). cap is None if no camera could be opened.
    """
    controller.release_all()
    runner_controller.release_all()
    grabber.stop()
    cap.release()

    cap, cam_index = open_camera(cam_index + 1)
    if cap is None:
        print("[ERROR] Could not open another camera.")
        return None, None, cam_index

    # Re-apply configuration and verify frames are actually arriving.
    _configure(cap)

    # Warm up the camera - some cameras (especially USB/DroidCam) need a few frames.
    print(f"[INFO] Warming up camera index {cam_index}...")
    for _ in range(10):
        ok, test_frame = cap.read()
        if ok and test_frame is not None:
            break
        time.sleep(0.05)

    ok, test_frame = cap.read()
    if not ok or test_frame is None:
        print(f"[WARN] Camera {cam_index} opened but delivers no frames. Trying next...")
        cap.release()
        return switch_camera(cap, FrameGrabber(cap) if cap else None, cam_index, controller, runner_controller)

    print(f"[INFO] Switched to camera index {cam_index}")
    new_fps = measure_fps(cap)
    print(f"[INFO] Capture: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ {new_fps:.0f} fps measured "
          f"({CAMERA_FPS} requested)")
    return cap, FrameGrabber(cap), cam_index


def main():
    cap, cam_index = open_camera(CAMERA_INDEX)
    if cap is None:
        print("[ERROR] Cannot open camera.")
        print("  -> macOS: System Settings > Privacy & Security > Camera")
        print("  -> DroidCam: open DroidCam Client on PC first, phone must show 'connected'")
        print("     try CAMERA_INDEX = 0, 1, 2... (DroidCam usually registers as the")
        print("     last index, after your built-in/USB webcams)")
        return

    # Format is applied in open_camera(), before the stream starts — do not re-set
    # it here, it would be ignored anyway.
    actual_fps = measure_fps(cap)
    print(f"[INFO] Capture: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ {actual_fps:.0f} fps measured "
          f"({CAMERA_FPS} requested)")
    if actual_fps < CAMERA_FPS * 0.8:
        print(f"[WARN] Camera is delivering {actual_fps:.0f} fps, which caps how fast")
        print("       gestures can register. In the DroidCam phone app raise the FPS")
        print("       limit, or use USB instead of WiFi.")

    setup_window()

    grabber           = FrameGrabber(cap)
    controller        = SteeringController()
    runner_controller = RunnerController()
    shifter           = PaddleShifter()
    game_mode         = GAME_MODE

    hands = make_hands(game_mode)

    conn_style     = mp_drawing.DrawingSpec(color=(80, 80, 100), thickness=1)
    landmark_style = mp_drawing.DrawingSpec(color=(200, 200, 255), thickness=1, circle_radius=2)

    prev_time     = time.time()
    angle         = 0.0
    direction     = "STRAIGHT"
    strength      = 0.0
    throttle_mode = "NEUTRAL"
    left_open     = False
    right_open    = False
    lost_frames   = 0
    runner_action = "READY"
    flip_camera   = FLIP_CAMERA

    print("=" * 55)
    print("  Virtual Steering Wheel  |  ESC=quit, M=mode, C=camera, F=mirror")
    print("=" * 55)
    print("  RACING: FIST=Accelerate  OPEN=Brake  Tilt=Steer")
    if PADDLE_ENABLED:
        print(f"          Paddles: flick 2 fingers  RIGHT='{GEAR_UP_KEY}' (up)  "
              f"LEFT='{GEAR_DOWN_KEY}' (down)")
    print("  RUNNER: hold up index+middle finger, swipe hand Left/Right/Up/Down")
    print(f"  Starting in {game_mode} mode")
    print("=" * 55)

    try:
        while True:
            frame = grabber.read()
            if frame is None:
                # No new frame yet; yield briefly rather than reprocessing a stale one.
                time.sleep(0.001)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC to quit (safer than q which conflicts with gear shift)
                    break
                if key in (ord('c'), ord('C')):
                    cap, grabber, cam_index = switch_camera(
                        cap, grabber, cam_index, controller, runner_controller)
                    if cap is None:
                        break
                    keep_window_on_top()
                    prev_time = time.time()
                if key in (ord('f'), ord('F')):
                    flip_camera = not flip_camera
                    print(f"[INFO] Mirror camera: {'ON' if flip_camera else 'OFF'}")
                continue

            if flip_camera:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            both_visible = False
            pose_active  = False
            pose_point   = None

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_data = {}

                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_style, conn_style)

                    wrist     = hand_landmarks.landmark[0]
                    wx        = int(wrist.x * w)
                    wy        = int(wrist.y * h)
                    opened    = is_open_hand(hand_landmarks)
                    two_finger = is_two_finger_pose(hand_landmarks)
                    hand_data[label] = (wrist.x, wrist.y, wx, wy, opened, two_finger)

                    if game_mode == "RUNNER" and not pose_active and two_finger:
                        idx_tip = hand_landmarks.landmark[8]
                        mid_tip = hand_landmarks.landmark[12]
                        px = (idx_tip.x + mid_tip.x) / 2.0
                        py = (idx_tip.y + mid_tip.y) / 2.0
                        pose_active = True
                        pose_point  = (px, py)
                        cv2.circle(frame, (int(px * w), int(py * h)), 10, CLR_ACCENT, -1)

                if game_mode == "RACING":
                    # Paddles run off whichever hands are visible, so a quick
                    # one-handed flick still shifts.
                    shifter.update({h: hand_data[h][5] for h in hand_data})

                    if "Left" in hand_data and "Right" in hand_data:
                        both_visible = True
                        lost_frames  = 0

                        lx_n, ly_n, lx_px, ly_px, left_open,  _ = hand_data["Left"]
                        rx_n, ry_n, rx_px, ry_px, right_open, _ = hand_data["Right"]

                        draw_hand_connection(frame, (lx_px, ly_px), (rx_px, ry_px))
                        angle, direction, strength = controller.update_steer((lx_n, ly_n), (rx_n, ry_n))
                        throttle_mode = controller.update_throttle(left_open, right_open)
                    else:
                        lost_frames += 1
                        if lost_frames >= GRACE_FRAMES:
                            controller.release_all()
                            angle, direction, strength = 0.0, "STRAIGHT", 0.0
                            throttle_mode = "NEUTRAL"
                            left_open = right_open = False
                else:
                    lost_frames   = 0
                    runner_action = runner_controller.update(pose_active, pose_point, w / h)
            else:
                lost_frames += 1
                shifter.update({})          # no hands: let both paddles re-arm
                if lost_frames >= GRACE_FRAMES:
                    controller.release_all()
                    runner_controller.release_all()
                    angle, direction, strength = 0.0, "STRAIGHT", 0.0
                    throttle_mode = "NEUTRAL"
                    runner_action = "READY"
                    left_open = right_open = False

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            if game_mode == "RACING":
                draw_hud(frame, angle, direction, strength, throttle_mode, both_visible,
                         left_open, right_open, fps,
                         gear=shifter.gear if PADDLE_ENABLED else None,
                         shift_flash=shifter.flash())
            else:
                if SHOW_SWIPE_TRAIL:
                    draw_swipe_trail(frame, runner_controller.trail(), w / h)
                draw_runner_hud(frame, runner_action, pose_active,
                                not runner_controller.needs_settle, fps)

            cv2.imshow(WINDOW_NAME, frame)
            keep_window_on_top()

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to quit (safer than q which conflicts with gear shift)
                break
            if key in (ord('m'), ord('M')):
                controller.release_all()
                runner_controller.release_all()
                shifter.reset()
                game_mode     = "RUNNER" if game_mode == "RACING" else "RACING"
                direction     = "STRAIGHT"
                strength      = 0.0
                throttle_mode = "NEUTRAL"
                runner_action = "READY"
                lost_frames   = 0
                hands.close()
                hands = make_hands(game_mode)
                print(f"[INFO] Switched to {game_mode} mode")
            if key in (ord('c'), ord('C')):
                cap, grabber, cam_index = switch_camera(
                    cap, grabber, cam_index, controller, runner_controller)
                if cap is None:
                    break
                prev_time = time.time()
            if key in (ord('f'), ord('F')):
                flip_camera = not flip_camera
                print(f"[INFO] Mirror camera: {'ON' if flip_camera else 'OFF'}")

    finally:
        controller.release_all()
        runner_controller.release_all()
        if grabber is not None:
            grabber.stop()
        hands.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Stopped. All keys released.")


if __name__ == "__main__":
    main()
