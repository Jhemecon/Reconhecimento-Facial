import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import urllib.request

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DETECTOR_PATH = MODEL_DIR / "detector.onnx"
RECOGNIZER_PATH = MODEL_DIR / "recognizer.onnx"
MEDIAPIPE_MODEL = MODEL_DIR / "face_landmarker.task"

DETECTOR_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
RECOGNIZER_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
MEDIAPIPE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

THRESHOLD = 0.6
MIN_FACE = 80
DETECT_CONF_THRESHOLD = 0.5

# MediaPipe landmark indices
class LM:
    # Eye landmarks (for EAR)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    LEFT_IRIS = 468
    RIGHT_IRIS = 473
    
    # Mouth landmarks (for MAR)
    MOUTH_OUTER = [61, 291, 13, 14, 17, 84, 181, 87, 91, 78, 308, 402, 317, 14, 317]
    MOUTH_INNER = [78, 308, 13, 14, 82, 312, 87, 317]
    
    # Head pose (solvePnP reference points)
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_CORNER = 33
    RIGHT_EYE_CORNER = 263
    LEFT_EYE_OUTER = 133
    RIGHT_EYE_OUTER = 362
    FOREHEAD = 10
    
    # Face oval for alignment
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def download_models():
    if not DETECTOR_PATH.exists():
        print("Baixando detector YuNet...")
        urllib.request.urlretrieve(DETECTOR_URL, DETECTOR_PATH)
    if not RECOGNIZER_PATH.exists():
        print("Baixando reconhecedor SFace...")
        urllib.request.urlretrieve(RECOGNIZER_URL, RECOGNIZER_PATH)
    if not MEDIAPIPE_MODEL.exists():
        print("Baixando MediaPipe FaceLandmarker...")
        urllib.request.urlretrieve(MEDIAPIPE_URL, MEDIAPIPE_MODEL)
    print("Modelos OK")


def eye_aspect_ratio(landmarks, eye_indices):
    """Calculate EAR using 6-point eye model"""
    # Vertical distances
    v1 = np.linalg.norm([landmarks[eye_indices[1]].x - landmarks[eye_indices[5]].x,
                         landmarks[eye_indices[1]].y - landmarks[eye_indices[5]].y,
                         landmarks[eye_indices[1]].z - landmarks[eye_indices[5]].z])
    v2 = np.linalg.norm([landmarks[eye_indices[2]].x - landmarks[eye_indices[4]].x,
                         landmarks[eye_indices[2]].y - landmarks[eye_indices[4]].y,
                         landmarks[eye_indices[2]].z - landmarks[eye_indices[4]].z])
    # Horizontal distance
    h = np.linalg.norm([landmarks[eye_indices[0]].x - landmarks[eye_indices[3]].x,
                        landmarks[eye_indices[0]].y - landmarks[eye_indices[3]].y,
                        landmarks[eye_indices[0]].z - landmarks[eye_indices[3]].z])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0


def mouth_aspect_ratio(landmarks, mouth_indices):
    """Calculate MAR using mouth landmarks"""
    # Vertical: upper lip to lower lip
    v1 = np.linalg.norm([landmarks[mouth_indices[2]].x - landmarks[mouth_indices[3]].x,
                         landmarks[mouth_indices[2]].y - landmarks[mouth_indices[3]].y,
                         landmarks[mouth_indices[2]].z - landmarks[mouth_indices[3]].z])
    # Horizontal: mouth corners
    h = np.linalg.norm([landmarks[mouth_indices[0]].x - landmarks[mouth_indices[1]].x,
                        landmarks[mouth_indices[0]].y - landmarks[mouth_indices[1]].y,
                        landmarks[mouth_indices[0]].z - landmarks[mouth_indices[1]].z])
    return v1 / h if h > 0 else 0


def head_pose_estimation(landmarks, frame_w, frame_h):
    """Estimate head pose using solvePnP with 3D model points"""
    # 3D model points (generic face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),           # Nose tip
        (0.0, -330.0, -65.0),      # Chin
        (-225.0, 170.0, -135.0),   # Left eye left corner
        (225.0, 170.0, -135.0),    # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),   # Right mouth corner
    ], dtype=np.float32)

    # 2D image points from landmarks
    image_points = np.array([
        (landmarks[LM.NOSE_TIP].x * frame_w, landmarks[LM.NOSE_TIP].y * frame_h),
        (landmarks[LM.CHIN].x * frame_w, landmarks[LM.CHIN].y * frame_h),
        (landmarks[LM.LEFT_EYE_CORNER].x * frame_w, landmarks[LM.LEFT_EYE_CORNER].y * frame_h),
        (landmarks[LM.RIGHT_EYE_CORNER].x * frame_w, landmarks[LM.RIGHT_EYE_CORNER].y * frame_h),
        (landmarks[LM.LEFT_EYE_OUTER].x * frame_w, landmarks[LM.LEFT_EYE_OUTER].y * frame_h),
        (landmarks[LM.RIGHT_EYE_OUTER].x * frame_w, landmarks[LM.RIGHT_EYE_OUTER].y * frame_h),
    ], dtype=np.float32)

    # Camera matrix
    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    # Convert rotation vector to Euler angles
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = euler_angles.flatten()[:3]
    return {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}


def iris_position(landmarks):
    """Get iris positions for gaze estimation"""
    left_iris = landmarks[LM.LEFT_IRIS]
    right_iris = landmarks[LM.RIGHT_IRIS]
    left_eye_corner = landmarks[LM.LEFT_EYE_CORNER]
    right_eye_corner = landmarks[LM.RIGHT_EYE_CORNER]
    left_eye_outer = landmarks[LM.LEFT_EYE_OUTER]
    right_eye_outer = landmarks[LM.RIGHT_EYE_OUTER]

    # Normalize iris position within eye
    left_eye_width = abs(left_eye_outer.x - left_eye_corner.x)
    right_eye_width = abs(right_eye_outer.x - right_eye_corner.x)

    left_gaze_x = (left_iris.x - left_eye_corner.x) / left_eye_width if left_eye_width > 0 else 0.5
    right_gaze_x = (right_iris.x - right_eye_corner.x) / right_eye_width if right_eye_width > 0 else 0.5

    return {"left": left_gaze_x, "right": right_gaze_x}


def face_quality_score(landmarks, frame_w, frame_h):
    """Compute face quality based on landmarks"""
    # Symmetry score
    left_eye = np.array([landmarks[LM.LEFT_EYE_CORNER].x, landmarks[LM.LEFT_EYE_CORNER].y])
    right_eye = np.array([landmarks[LM.RIGHT_EYE_CORNER].x, landmarks[LM.RIGHT_EYE_CORNER].y])
    nose = np.array([landmarks[LM.NOSE_TIP].x, landmarks[LM.NOSE_TIP].y])

    # Distance from nose to midpoint of eyes
    eye_mid = (left_eye + right_eye) / 2
    symmetry = 1.0 - min(1.0, np.linalg.norm(nose - eye_mid) * 2)

    # Face size relative to frame
    face_w = abs(right_eye[0] - left_eye[0]) * frame_w
    face_h = abs(landmarks[LM.CHIN].y - landmarks[LM.FOREHEAD].y) * frame_h
    size_score = min(1.0, (face_w * face_h) / (frame_w * frame_h * 0.15))

    # Visibility (how many landmarks are confident)
    # MediaPipe doesn't give confidence per landmark, use z-depth as proxy
    z_values = [lm.z for lm in landmarks]
    visibility = 1.0 - min(1.0, abs(np.mean(z_values)) * 10)

    return {
        "symmetry": float(symmetry),
        "size": float(size_score),
        "visibility": float(visibility),
        "overall": float((symmetry + size_score + visibility) / 3)
    }


class FaceEngine:
    def __init__(self):
        # YuNet detector (fast, for initial detection)
        self.detector = cv2.dnn.readNet(str(DETECTOR_PATH))
        self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # SFace recognizer (for embeddings)
        self.recognizer = cv2.dnn.readNet(str(RECOGNIZER_PATH))
        self.recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # MediaPipe FaceLandmarker (for detailed analysis)
        base_options = python.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # Previous frame for temporal smoothing
        self.prev_landmarks = None
        self.prev_embedding = None

    def detect(self, frame):
        """Fast detection with YuNet, returns the best face bounding box"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320), (0, 0, 0), swapRB=True, crop=False)
        self.detector.setInput(blob)
        outs = self.detector.forward(self.detector.getUnconnectedOutLayers().flatten().tolist())

        boxes = []
        confs = []
        strides = [8, 16, 32]
        scale_x = w / 320.0
        scale_y = h / 320.0

        for idx, stride in enumerate(strides):
            conf_map = outs[idx].reshape(-1, 1)
            bbox_map = outs[idx + 6].reshape(-1, 4)

            fm_w = 320 // stride
            fm_h = 320 // stride

            for i, conf in enumerate(conf_map):
                if conf[0] < DETECT_CONF_THRESHOLD:
                    continue

                cx_off, cy_off, bw, bh = bbox_map[i]
                cell_x = i % fm_w
                cell_y = i // fm_w
                cell_cx = (cell_x + 0.5) * stride
                cell_cy = (cell_y + 0.5) * stride

                abs_cx = (cell_cx + cx_off * stride) * scale_x
                abs_cy = (cell_cy + cy_off * stride) * scale_y
                abs_w = bw * stride * scale_x
                abs_h = bh * stride * scale_y

                x1 = int(abs_cx - abs_w / 2)
                y1 = int(abs_cy - abs_h / 2)
                x2 = int(abs_cx + abs_w / 2)
                y2 = int(abs_cy + abs_h / 2)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 < MIN_FACE or y2 - y1 < MIN_FACE:
                    continue

                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confs.append(float(conf[0]))

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confs, DETECT_CONF_THRESHOLD, 0.3)
        if len(indices) == 0:
            return []

        faces = []
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            faces.append((x, y, x + bw, y + bh, confs[i]))
        
        # Sort by confidence, return only the best face
        faces.sort(key=lambda f: f[4], reverse=True)
        return [faces[0]] if faces else []

    def analyze(self, frame, face_box=None):
        """Full face analysis using MediaPipe FaceLandmarker"""
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect landmarks
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        h, w = frame.shape[:2]

        # Extract all metrics
        left_ear = eye_aspect_ratio(landmarks, LM.LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, LM.RIGHT_EYE)
        ear = (left_ear + right_ear) / 2

        mar = mouth_aspect_ratio(landmarks, LM.MOUTH_INNER)

        pose = head_pose_estimation(landmarks, w, h)

        gaze = iris_position(landmarks)

        quality = face_quality_score(landmarks, w, h)

        # Blendshapes (facial expressions)
        blendshapes = {}
        if result.face_blendshapes:
            for bs in result.face_blendshapes[0]:
                blendshapes[bs.category_name] = bs.score

        # Temporal smoothing
        if self.prev_landmarks is not None:
            # Smooth landmarks
            alpha = 0.3
            for i, lm in enumerate(landmarks):
                prev = self.prev_landmarks[i]
                lm.x = alpha * lm.x + (1 - alpha) * prev.x
                lm.y = alpha * lm.y + (1 - alpha) * prev.y
                lm.z = alpha * lm.z + (1 - alpha) * prev.z

        self.prev_landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks]

        return {
            "landmarks": landmarks,
            "ear": ear,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "mar": mar,
            "pose": pose,
            "gaze": gaze,
            "quality": quality,
            "blendshapes": blendshapes,
            "bbox": face_box,
        }

    def align(self, frame, landmarks):
        """Align face using eye landmarks"""
        left_eye = landmarks[LM.LEFT_EYE_CORNER]
        right_eye = landmarks[LM.RIGHT_EYE_CORNER]

        h, w = frame.shape[:2]
        left_pt = (int(left_eye.x * w), int(left_eye.y * h))
        right_pt = (int(right_eye.x * w), int(right_eye.y * h))

        dy = right_pt[1] - left_pt[1]
        dx = right_pt[0] - left_pt[0]
        angle = np.degrees(np.arctan2(dy, dx))
        center = ((left_pt[0] + right_pt[0]) / 2, (left_pt[1] + right_pt[1]) / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += 56 - center[0]
        M[1, 2] += 39 - center[1]

        return cv2.warpAffine(frame, M, (112, 112))

    def embed(self, aligned):
        """Generate SFace embedding"""
        blob = cv2.dnn.blobFromImage(aligned, 1 / 255.0, (112, 112), (0, 0, 0), swapRB=True)
        self.recognizer.setInput(blob)
        emb = self.recognizer.forward().flatten()
        emb = emb / np.linalg.norm(emb)

        # Temporal smoothing
        if self.prev_embedding is not None:
            emb = 0.7 * emb + 0.3 * self.prev_embedding
            emb = emb / np.linalg.norm(emb)
        self.prev_embedding = emb

        return emb

    @staticmethod
    def match(e1, e2):
        return np.dot(e1, e2) >= THRESHOLD


# Legacy compatibility
engine = FaceEngine()