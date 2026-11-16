import cv2
import numpy as np
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import urllib.request
import os

# ===== CONFIG =====
DB = "attendance.db"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DETECTOR_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
RECOGNIZER_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
DETECTOR_PATH = MODEL_DIR / "detector.onnx"
RECOGNIZER_PATH = MODEL_DIR / "recognizer.onnx"
THRESHOLD = 0.6
MIN_FACE = 80
DETECT_CONF_THRESHOLD = 0.5

# ===== DOWNLOAD MODELS =====
def download_models():
    if not DETECTOR_PATH.exists():
        print("Baixando detector...")
        urllib.request.urlretrieve(DETECTOR_URL, DETECTOR_PATH)
    if not RECOGNIZER_PATH.exists():
        print("Baixando reconhecedor...")
        urllib.request.urlretrieve(RECOGNIZER_URL, RECOGNIZER_PATH)
    print("Modelos OK")

# ===== DATABASE =====
def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            class_id TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            vector BLOB NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)
    conn.commit()
    conn.close()

def get_conn():
    return sqlite3.connect(DB)

# ===== FACE ENGINE =====
class FaceEngine:
    def __init__(self):
        self.detector = cv2.dnn.readNet(str(DETECTOR_PATH))
        self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.recognizer = cv2.dnn.readNet(str(RECOGNIZER_PATH))
        self.recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320), (0, 0, 0), swapRB=True, crop=False)
        self.detector.setInput(blob)
        outs = self.detector.forward(self.detector.getUnconnectedOutLayers().flatten().tolist())
        
        boxes = []
        confs = []
        kps_list = []
        strides = [8, 16, 32]
        scale_x = w / 320.0
        scale_y = h / 320.0
        
        for idx, stride in enumerate(strides):
            conf_map = outs[idx].reshape(-1, 1)
            bbox_map = outs[idx + 6].reshape(-1, 4)
            kps_map = outs[idx + 9].reshape(-1, 10)
            
            fm_w = 320 // stride
            fm_h = 320 // stride
            
            for i, conf in enumerate(conf_map):
                if conf[0] < DETECT_CONF_THRESHOLD: continue
                
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
                if x2 - x1 < MIN_FACE or y2 - y1 < MIN_FACE: continue
                
                kps = kps_map[i].reshape(-1, 2)
                kps[:, 0] = (kps[:, 0] * stride * scale_x).astype(int)
                kps[:, 1] = (kps[:, 1] * stride * scale_y).astype(int)
                
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confs.append(float(conf[0]))
                kps_list.append(kps)
        
        if not boxes:
            return []
        
        indices = cv2.dnn.NMSBoxes(boxes, confs, DETECT_CONF_THRESHOLD, 0.4)
        if len(indices) == 0:
            return []
        
        faces = []
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            faces.append((x, y, x + bw, y + bh, kps_list[i]))
        return faces
    
    def align(self, frame, lm):
        left, right = lm[0], lm[1]
        angle = np.degrees(np.arctan2(right[1]-left[1], right[0]-left[0]))
        center = ((left[0]+right[0])/2, (left[1]+right[1])/2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0,2] += 56 - center[0]; M[1,2] += 39 - center[1]
        return cv2.warpAffine(frame, M, (112, 112))
    
    def embed(self, aligned):
        blob = cv2.dnn.blobFromImage(aligned, 1/255.0, (112, 112), (0,0,0), swapRB=True)
        self.recognizer.setInput(blob)
        emb = self.recognizer.forward().flatten()
        return emb / np.linalg.norm(emb)
    
    @staticmethod
    def match(e1, e2): return np.dot(e1, e2) >= THRESHOLD

# ===== FASTAPI =====
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class StudentIn(BaseModel):
    code: str
    name: str
    class_id: str = ""

# Startup
download_models()
init_db()
engine = FaceEngine()

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/students")
def create_student(s: StudentIn):
    conn = get_conn()
    try:
        cur = conn.execute("INSERT INTO students (code, name, class_id) VALUES (?, ?, ?)", (s.code, s.name, s.class_id))
        conn.commit()
        return {"id": cur.lastrowid, "code": s.code, "name": s.name}
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Código já existe")
    finally:
        conn.close()

@app.get("/students")
def list_students(class_id: str = ""):
    conn = get_conn()
    if class_id:
        rows = conn.execute("SELECT * FROM students WHERE class_id = ?", (class_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM students").fetchall()
    conn.close()
    return [{"id": r[0], "code": r[1], "name": r[2], "class_id": r[3]} for r in rows]

@app.post("/students/{code}/face")
async def register_face(code: str, file: UploadFile = File(...)):
    conn = get_conn()
    student = conn.execute("SELECT * FROM students WHERE code = ?", (code,)).fetchone()
    conn.close()
    if not student: raise HTTPException(404, "Aluno não encontrado")
    
    data = await file.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    faces = engine.detect(frame)
    if not faces: raise HTTPException(400, "Nenhum rosto detectado")
    
    x1, y1, x2, y2, lm = faces[0]
    aligned = engine.align(frame, lm)
    emb = engine.embed(aligned)
    
    conn = get_conn()
    conn.execute("INSERT INTO embeddings (student_id, vector) VALUES (?, ?)", 
                 (student[0], emb.astype(np.float32).tobytes()))
    conn.commit()
    conn.close()
    return {"message": "Rosto cadastrado"}

@app.post("/attendance")
async def mark_attendance(code: str = Form(...), type: str = Form(...)):
    if type not in ("entry", "exit"): raise HTTPException(400, "Tipo: entry ou exit")
    conn = get_conn()
    student = conn.execute("SELECT * FROM students WHERE code = ?", (code,)).fetchone()
    if not student: raise HTTPException(404, "Aluno não encontrado")
    conn.execute("INSERT INTO attendance (student_id, timestamp, type) VALUES (?, ?, ?)",
                 (student[0], datetime.now().isoformat(), type))
    conn.commit()
    conn.close()
    return {"message": f"Ponto {type} registrado"}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    data = await file.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"match": False, "error": "Imagem inválida"}
    faces = engine.detect(frame)
    if not faces:
        return {"match": False}
    
    _, _, _, _, lm = faces[0]
    emb = engine.embed(engine.align(frame, lm))
    
    conn = get_conn()
    students = conn.execute("SELECT * FROM students").fetchall()
    for s in students:
        rows = conn.execute("SELECT vector FROM embeddings WHERE student_id = ?", (s[0],)).fetchall()
        for r in rows:
            stored = np.frombuffer(r[0], np.float32)
            if engine.match(emb, stored):
                conn.execute("INSERT INTO attendance (student_id, timestamp, type) VALUES (?, ?, ?)",
                             (s[0], datetime.now().isoformat(), "entry"))
                conn.commit()
                conn.close()
                return {"match": True, "student": {"code": s[1], "name": s[2]}}
    conn.close()
    return {"match": False}

@app.get("/attendance/{code}")
def get_attendance(code: str):
    conn = get_conn()
    student = conn.execute("SELECT * FROM students WHERE code = ?", (code,)).fetchone()
    conn.close()
    if not student: raise HTTPException(404, "Não encontrado")
    conn = get_conn()
    rows = conn.execute("SELECT timestamp, type FROM attendance WHERE student_id = ? ORDER BY timestamp", (student[0],)).fetchall()
    conn.close()
    return [{"timestamp": r[0], "type": r[1]} for r in rows]

@app.get("/report/{class_id}")
def class_report(class_id: str, date: str):
    conn = get_conn()
    rows = conn.execute("""
        SELECT s.code, s.name,
               MIN(CASE WHEN a.type='entry' THEN a.timestamp END) as entry,
               MAX(CASE WHEN a.type='exit' THEN a.timestamp END) as exit
        FROM students s
        LEFT JOIN attendance a ON s.id = a.student_id AND date(a.timestamp) = ?
        WHERE s.class_id = ?
        GROUP BY s.id
    """, (date, class_id)).fetchall()
    conn.close()
    return [{"code": r[0], "name": r[1], "entry": r[2], "exit": r[3]} for r in rows]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)