import cv2
import numpy as np
import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from pathlib import Path
import threading
import urllib.request
import os
import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.face_engine_v2 import FaceEngine as FaceEngineV2, download_models, THRESHOLD, MIN_FACE, DETECT_CONF_THRESHOLD, DETECTOR_PATH, RECOGNIZER_PATH, MEDIAPIPE_MODEL

# ===== CONFIG =====
DB = "attendance.db"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DETECTOR_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
RECOGNIZER_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
MEDIAPIPE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DETECTOR_PATH = MODEL_DIR / "detector.onnx"
RECOGNIZER_PATH = MODEL_DIR / "recognizer.onnx"
MEDIAPIPE_MODEL = MODEL_DIR / "face_landmarker.task"

THRESHOLD = 0.6
MIN_FACE = 80
DETECT_CONF_THRESHOLD = 0.5

# ===== DOWNLOAD MODELS =====
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

# ===== WRAPPER FACE ENGINE =====
class FaceEngine:
    """Wrapper que usa FaceEngineV2 (MediaPipe) para analise completa"""
    def __init__(self):
        self.v2 = FaceEngineV2()
    
    def detect(self, frame):
        """Retorna formato compativel: (x1, y1, x2, y2, conf) - apenas melhor face"""
        faces = self.v2.detect(frame)
        if not faces:
            return []
        return faces
    
    def analyze(self, frame, face_box):
        """Analise completa com MediaPipe"""
        return self.v2.analyze(frame, face_box)
    
    def align(self, frame, landmarks):
        return self.v2.align(frame, landmarks)
    
    def embed(self, aligned):
        return self.v2.embed(aligned)
    
    @staticmethod
    def match(e1, e2):
        return np.dot(e1, e2) >= THRESHOLD

# ===== TKINTER APP =====
class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Ponto Escolar - Reconhecimento Facial")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        download_models()
        init_db()
        self.engine = FaceEngine()

        self.cap = None
        self.camera_running = False
        self.current_frame = None
        self.recognition_mode = False
        
        # Recognition thread
        self.recognition_thread = None
        self.recognition_active = False
        self.latest_analysis = None
        self.recognition_frame = None
        self.frame_count = 0

        self.setup_ui()
        self.load_students()

    def setup_ui(self):
        # Notebook (abas)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Aba 1: Camera / Reconhecimento
        self.tab_camera = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_camera, text="Camera & Reconhecimento")
        self.setup_camera_tab()

        # Aba 2: Alunos
        self.tab_students = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_students, text="Alunos")
        self.setup_students_tab()

        # Aba 3: Ponto / Frequencia
        self.tab_attendance = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_attendance, text="Frequencia")
        self.setup_attendance_tab()

        # Aba 4: Relatorios
        self.tab_reports = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_reports, text="Relatorios")
        self.setup_reports_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Pronto")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)

    def setup_camera_tab(self):
        # Left: Video feed
        left_frame = ttk.LabelFrame(self.tab_camera, text="Camera")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Container para o video com tamanho fixo
        video_container = ttk.Frame(left_frame)
        video_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        video_container.pack_propagate(False)

        self.video_label = ttk.Label(video_container, text="Camera parada\n\nClique em 'Iniciar Camera'", anchor=tk.CENTER, font=('Arial', 12))
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Camera controls
        cam_controls = ttk.Frame(left_frame)
        cam_controls.pack(fill=tk.X, padx=10, pady=5)

        self.btn_start_camera = ttk.Button(cam_controls, text="Iniciar Camera", command=self.start_camera)
        self.btn_start_camera.pack(side=tk.LEFT, padx=5)

        self.btn_stop_camera = ttk.Button(cam_controls, text="Parar Camera", command=self.stop_camera, state=tk.DISABLED)
        self.btn_stop_camera.pack(side=tk.LEFT, padx=5)

        self.btn_recognize = ttk.Button(cam_controls, text="Modo Reconhecimento", command=self.toggle_recognition, state=tk.DISABLED)
        self.btn_recognize.pack(side=tk.LEFT, padx=5)

        # Right: Controls & Log
        right_frame = ttk.Frame(self.tab_camera)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Manual attendance
        manual_frame = ttk.LabelFrame(right_frame, text="Ponto Manual")
        manual_frame.pack(fill=tk.X, pady=5)

        ttk.Label(manual_frame, text="Codigo do Aluno:").pack(anchor=tk.W, padx=5, pady=2)
        self.manual_code_var = tk.StringVar()
        ttk.Entry(manual_frame, textvariable=self.manual_code_var).pack(fill=tk.X, padx=5, pady=2)

        btn_frame = ttk.Frame(manual_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(btn_frame, text="Entrada", command=lambda: self.manual_attendance("entry")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Saida", command=lambda: self.manual_attendance("exit")).pack(side=tk.LEFT, padx=2)

        # Register face
        reg_frame = ttk.LabelFrame(right_frame, text="Cadastrar Rosto")
        reg_frame.pack(fill=tk.X, pady=5)

        ttk.Label(reg_frame, text="Codigo do Aluno:").pack(anchor=tk.W, padx=5, pady=2)
        self.reg_code_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.reg_code_var).pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(reg_frame, text="Capturar da Camera", command=self.capture_face).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(reg_frame, text="Carregar Imagem", command=self.load_face_image).pack(fill=tk.X, padx=5, pady=2)
        ttk.Separator(reg_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(reg_frame, text="+ Novo Aluno (sem camera)", command=self.new_student_dialog).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(reg_frame, text="+ Registrar Aluno + Rosto (camera)", command=self.register_unknown_student).pack(fill=tk.X, padx=5, pady=2)

        # Log
        log_frame = ttk.LabelFrame(right_frame, text="Log de Eventos")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = tk.Text(log_frame, height=15, width=40)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

    def setup_students_tab(self):
        # Toolbar
        toolbar = ttk.Frame(self.tab_students)
        toolbar.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(toolbar, text="Novo Aluno", command=self.new_student_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Editar", command=self.edit_student_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Excluir", command=self.delete_student).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Cadastrar Rosto", command=self.register_face_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Atualizar", command=self.load_students).pack(side=tk.LEFT, padx=5)

        # Treeview
        columns = ("ID", "Codigo", "Nome", "Turma", "Faces Cadastradas")
        self.students_tree = ttk.Treeview(self.tab_students, columns=columns, show="headings")
        for col in columns:
            self.students_tree.heading(col, text=col)
            self.students_tree.column(col, width=150)
        self.students_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Scrollbar
        scrollbar = ttk.Scrollbar(self.tab_students, orient=tk.VERTICAL, command=self.students_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.students_tree.config(yscrollcommand=scrollbar.set)

    def setup_attendance_tab(self):
        # Filter
        filter_frame = ttk.Frame(self.tab_attendance)
        filter_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(filter_frame, text="Codigo:").pack(side=tk.LEFT, padx=5)
        self.att_code_var = tk.StringVar()
        ttk.Entry(filter_frame, textvariable=self.att_code_var, width=20).pack(side=tk.LEFT, padx=5)

        ttk.Label(filter_frame, text="Data Inicio:").pack(side=tk.LEFT, padx=5)
        self.att_start_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(filter_frame, textvariable=self.att_start_var, width=12).pack(side=tk.LEFT, padx=5)

        ttk.Label(filter_frame, text="Data Fim:").pack(side=tk.LEFT, padx=5)
        self.att_end_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(filter_frame, textvariable=self.att_end_var, width=12).pack(side=tk.LEFT, padx=5)

        ttk.Button(filter_frame, text="Filtrar", command=self.load_attendance).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame, text="Limpar", command=self.clear_attendance_filter).pack(side=tk.LEFT, padx=5)

        # Treeview
        columns = ("ID", "Aluno", "Codigo", "Data/Hora", "Tipo")
        self.att_tree = ttk.Treeview(self.tab_attendance, columns=columns, show="headings")
        for col in columns:
            self.att_tree.heading(col, text=col)
            self.att_tree.column(col, width=180)
        self.att_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = ttk.Scrollbar(self.tab_attendance, orient=tk.VERTICAL, command=self.att_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.att_tree.config(yscrollcommand=scrollbar.set)

    def setup_reports_tab(self):
        # Filter
        filter_frame = ttk.Frame(self.tab_reports)
        filter_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(filter_frame, text="Turma:").pack(side=tk.LEFT, padx=5)
        self.report_class_var = tk.StringVar()
        ttk.Entry(filter_frame, textvariable=self.report_class_var, width=15).pack(side=tk.LEFT, padx=5)

        ttk.Label(filter_frame, text="Data:").pack(side=tk.LEFT, padx=5)
        self.report_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(filter_frame, textvariable=self.report_date_var, width=12).pack(side=tk.LEFT, padx=5)

        ttk.Button(filter_frame, text="Gerar Relatorio", command=self.generate_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame, text="Exportar CSV", command=self.export_report_csv).pack(side=tk.LEFT, padx=5)

        # Treeview
        columns = ("Codigo", "Nome", "Entrada", "Saida", "Status")
        self.report_tree = ttk.Treeview(self.tab_reports, columns=columns, show="headings")
        for col in columns:
            self.report_tree.heading(col, text=col)
            self.report_tree.column(col, width=200)
        self.report_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = ttk.Scrollbar(self.tab_reports, orient=tk.VERTICAL, command=self.report_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.report_tree.config(yscrollcommand=scrollbar.set)

    # ===== CAMERA METHODS =====
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Erro", "Nao foi possivel acessar a camera")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera_running = True
        self.btn_start_camera.config(state=tk.DISABLED)
        self.btn_stop_camera.config(state=tk.NORMAL)
        self.btn_recognize.config(state=tk.NORMAL)
        self.update_video()

    def stop_camera(self):
        self.camera_running = False
        self.recognition_mode = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_start_camera.config(state=tk.NORMAL)
        self.btn_stop_camera.config(state=tk.DISABLED)
        self.btn_recognize.config(state=tk.DISABLED)
        self.video_label.config(image="", text="Camera parada")
        self.status_var.set("Camera parada")

    def toggle_recognition(self):
        self.recognition_mode = not self.recognition_mode
        if self.recognition_mode:
            self.btn_recognize.config(text="Parar Reconhecimento")
            self.status_var.set("Modo reconhecimento ATIVO")
            self.start_recognition_thread()
        else:
            self.btn_recognize.config(text="Modo Reconhecimento")
            self.status_var.set("Modo reconhecimento INATIVO")
            self.stop_recognition_thread()

    def start_recognition_thread(self):
        self.recognition_active = True
        self.recognition_thread = threading.Thread(target=self.recognition_worker, daemon=True)
        self.recognition_thread.start()

    def stop_recognition_thread(self):
        self.recognition_active = False
        if self.recognition_thread:
            self.recognition_thread.join(timeout=1.0)
            self.recognition_thread = None

    def recognition_worker(self):
        """Background thread for heavy MediaPipe processing"""
        frame_count = 0
        while self.recognition_active and self.camera_running:
            if self.current_frame is not None:
                frame_count += 1
                # Run analysis every 3 frames (~10 FPS) to avoid lag
                if frame_count % 3 == 0:
                    frame = self.current_frame.copy()
                    faces = self.engine.detect(frame)
                    if faces:
                        analysis = self.engine.analyze(frame, faces[0])
                        if analysis:
                            self.latest_analysis = analysis
                            self.recognition_frame = frame.copy()
            # Small sleep to not hog CPU
            threading.Event().wait(0.05)

    def update_video(self):
        if not self.camera_running or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame = frame.copy()
        display_frame = frame.copy()

        if self.recognition_mode:
            # Fast YuNet detection every frame
            faces = self.engine.detect(frame)
            if faces:
                x1, y1, x2, y2, conf = faces[0]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Use latest analysis from background thread
                if self.latest_analysis:
                    analysis = self.latest_analysis
                    landmarks = analysis["landmarks"]
                    h, w = frame.shape[:2]
                    
                    # Draw facial landmarks (every 3rd for performance)
                    for lm in landmarks[::3]:
                        px = int(lm.x * w)
                        py = int(lm.y * h)
                        cv2.circle(display_frame, (px, py), 1, (0, 255, 255), -1)

                    # Draw key points
                    key_indices = [33, 263, 1, 13, 14, 61, 291, 468, 473, 152, 10]
                    for idx in key_indices:
                        lm = landmarks[idx]
                        px = int(lm.x * w)
                        py = int(lm.y * h)
                        cv2.circle(display_frame, (px, py), 3, (255, 0, 255), -1)

                    # Display metrics
                    y_offset = y2 + 20
                    cv2.putText(display_frame, f"EAR: {analysis['ear']:.3f}", (x1, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y_offset += 18
                    cv2.putText(display_frame, f"MAR: {analysis['mar']:.3f}", (x1, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y_offset += 18
                    if analysis['pose']:
                        cv2.putText(display_frame, f"Pose: Y:{analysis['pose']['yaw']:.1f} P:{analysis['pose']['pitch']:.1f}", (x1, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y_offset += 18
                    cv2.putText(display_frame, f"Quality: {analysis['quality']['overall']:.2f}", (x1, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    # Try recognition using SFace (every 5 frames)
                    self.frame_count += 1
                    if self.frame_count % 5 == 0:
                        aligned = self.engine.align(frame, landmarks)
                        emb = self.engine.embed(aligned)
                        student = self.recognize_face(emb)
                        if student:
                            cv2.putText(display_frame, f"{student['name']}", (x1, y2 + 25),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            self.log_event(f"Reconhecido: {student['name']} ({student['code']})")
                            self.mark_attendance_auto(student['id'])

        # Convert to tkinter
        try:
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (640, 480))
            from PIL import Image, ImageTk
            photo = ImageTk.PhotoImage(Image.fromarray(img))
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo
        except Exception as e:
            self.status_var.set(f"Erro video: {e}")
            print(f"Video error: {e}")

        self.root.after(30, self.update_video)

    def recognize_face(self, emb):
        conn = get_conn()
        students = conn.execute("SELECT * FROM students").fetchall()
        for s in students:
            rows = conn.execute("SELECT vector FROM embeddings WHERE student_id = ?", (s[0],)).fetchall()
            for r in rows:
                stored = np.frombuffer(r[0], np.float32)
                if self.engine.match(emb, stored):
                    conn.close()
                    return {"id": s[0], "code": s[1], "name": s[2]}
        conn.close()
        return None

    def mark_attendance_auto(self, student_id):
        conn = get_conn()
        # Check last attendance to avoid duplicate
        last = conn.execute("""
            SELECT type FROM attendance WHERE student_id = ? ORDER BY timestamp DESC LIMIT 1
        """, (student_id,)).fetchone()
        next_type = "exit" if last and last[0] == "entry" else "entry"
        conn.execute("INSERT INTO attendance (student_id, timestamp, type) VALUES (?, ?, ?)",
                     (student_id, datetime.now().isoformat(), next_type))
        conn.commit()
        conn.close()
        self.log_event(f"Ponto {next_type} automatico para aluno ID {student_id}")

    def capture_face(self):
        code = self.reg_code_var.get().strip()
        if not code:
            messagebox.showwarning("Aviso", "Digite o codigo do aluno")
            return
        if self.current_frame is None:
            messagebox.showwarning("Aviso", "Camera nao iniciada")
            return

        conn = get_conn()
        student = conn.execute("SELECT * FROM students WHERE code = ?", (code,)).fetchone()
        conn.close()
        if not student:
            messagebox.showwarning("Aviso", "Aluno nao encontrado")
            return

        faces = self.engine.detect(self.current_frame)
        if not faces:
            messagebox.showwarning("Aviso", "Nenhum rosto detectado")
            return

        x1, y1, x2, y2, conf = faces[0]
        # Get landmarks via MediaPipe analysis
        analysis = self.engine.analyze(self.current_frame, faces[0])
        if not analysis:
            messagebox.showwarning("Aviso", "Nao foi possivel analisar o rosto")
            return

        aligned = self.engine.align(self.current_frame, analysis["landmarks"])
        emb = self.engine.embed(aligned)

        conn = get_conn()
        conn.execute("INSERT INTO embeddings (student_id, vector) VALUES (?, ?)",
                     (student[0], emb.astype(np.float32).tobytes()))
        conn.commit()
        conn.close()

        self.log_event(f"Rosto cadastrado para {student[2]} ({code})")
        messagebox.showinfo("Sucesso", "Rosto cadastrado com sucesso!")
        self.load_students()

    def load_face_image(self):
        code = self.reg_code_var.get().strip()
        if not code:
            messagebox.showwarning("Aviso", "Digite o codigo do aluno")
            return

        conn = get_conn()
        student = conn.execute("SELECT * FROM students WHERE code = ?", (code,)).fetchone()
        conn.close()
        if not student:
            messagebox.showwarning("Aviso", "Aluno nao encontrado")
            return

        path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.jpeg *.png")])
        if not path:
            return

        frame = cv2.imread(path)
        faces = self.engine.detect(frame)
        if not faces:
            messagebox.showwarning("Aviso", "Nenhum rosto detectado na imagem")
            return

        x1, y1, x2, y2, conf = faces[0]
        analysis = self.engine.analyze(frame, faces[0])
        if not analysis:
            messagebox.showwarning("Aviso", "Nao foi possivel analisar o rosto")
            return

        aligned = self.engine.align(frame, analysis["landmarks"])
        emb = self.engine.embed(aligned)

        conn = get_conn()
        conn.execute("INSERT INTO embeddings (student_id, vector) VALUES (?, ?)",
                     (student[0], emb.astype(np.float32).tobytes()))
        conn.commit()
        conn.close()

        self.log_event(f"Rosto carregado da imagem para {student[2]} ({code})")
        messagebox.showinfo("Sucesso", "Rosto cadastrado com sucesso!")
        self.load_students()

    def register_unknown_student(self):
        """Registra um aluno novo na hora: pede nome, codigo, turma e captura o rosto"""
        if self.current_frame is None:
            messagebox.showwarning("Aviso", "Inicie a camera primeiro")
            return

        # Detecta rosto primeiro
        faces = self.engine.detect(self.current_frame)
        if not faces:
            messagebox.showwarning("Aviso", "Nenhum rosto detectado na camera")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Registrar Novo Aluno")
        dialog.geometry("350x250")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Codigo do Aluno:").pack(anchor=tk.W, padx=10, pady=5)
        code_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=code_var).pack(fill=tk.X, padx=10)

        ttk.Label(dialog, text="Nome Completo:").pack(anchor=tk.W, padx=10, pady=5)
        name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=name_var).pack(fill=tk.X, padx=10)

        ttk.Label(dialog, text="Turma:").pack(anchor=tk.W, padx=10, pady=5)
        class_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=class_var).pack(fill=tk.X, padx=10)

        def save_new_student():
            code = code_var.get().strip()
            name = name_var.get().strip()
            class_id = class_var.get().strip()
            if not code or not name:
                messagebox.showwarning("Aviso", "Codigo e nome sao obrigatorios")
                return

            conn = get_conn()
            try:
                cur = conn.execute("INSERT INTO students (code, name, class_id) VALUES (?, ?, ?)",
                                   (code, name, class_id))
                student_id = cur.lastrowid
                conn.commit()
            except sqlite3.IntegrityError:
                messagebox.showerror("Erro", "Codigo ja existe")
                conn.close()
                return
            conn.close()

            # Captura e salva o rosto
            x1, y1, x2, y2, conf = faces[0]
            analysis = self.engine.analyze(self.current_frame, faces[0])
            if not analysis:
                messagebox.showwarning("Aviso", "Nao foi possivel analisar o rosto")
                return

            aligned = self.engine.align(self.current_frame, analysis["landmarks"])
            emb = self.engine.embed(aligned)

            conn = get_conn()
            conn.execute("INSERT INTO embeddings (student_id, vector) VALUES (?, ?)",
                         (student_id, emb.astype(np.float32).tobytes()))
            conn.commit()
            conn.close()

            self.log_event(f"Novo aluno registrado: {name} ({code}) - Turma: {class_id}")
            messagebox.showinfo("Sucesso", f"Aluno {name} registrado com rosto!")
            dialog.destroy()
            self.load_students()
            self.reg_code_var.set(code)

        ttk.Button(dialog, text="Salvar e Cadastrar Rosto", command=save_new_student).pack(pady=15)

    def manual_attendance(self, type_):
        code = self.manual_code_var.get().strip()
        if not code:
            messagebox.showwarning("Aviso", "Digite o codigo do aluno")
            return

        conn = get_conn()
        student = conn.execute("SELECT * FROM students WHERE code = ?", (code,)).fetchone()
        if not student:
            conn.close()
            messagebox.showwarning("Aviso", "Aluno nao encontrado")
            return

        conn.execute("INSERT INTO attendance (student_id, timestamp, type) VALUES (?, ?, ?)",
                     (student[0], datetime.now().isoformat(), type_))
        conn.commit()
        conn.close()

        self.log_event(f"Ponto manual {type_}: {student[2]} ({code})")
        messagebox.showinfo("Sucesso", f"Ponto {type_} registrado!")
        self.manual_code_var.set("")
        self.load_attendance()

    # ===== STUDENTS METHODS =====
    def load_students(self):
        for item in self.students_tree.get_children():
            self.students_tree.delete(item)

        conn = get_conn()
        students = conn.execute("SELECT * FROM students").fetchall()
        for s in students:
            face_count = conn.execute("SELECT COUNT(*) FROM embeddings WHERE student_id = ?", (s[0],)).fetchone()[0]
            self.students_tree.insert("", tk.END, values=(s[0], s[1], s[2], s[3], face_count))
        conn.close()

    def new_student_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Novo Aluno")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Codigo:").pack(anchor=tk.W, padx=10, pady=5)
        code_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=code_var).pack(fill=tk.X, padx=10)

        ttk.Label(dialog, text="Nome:").pack(anchor=tk.W, padx=10, pady=5)
        name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=name_var).pack(fill=tk.X, padx=10)

        ttk.Label(dialog, text="Turma:").pack(anchor=tk.W, padx=10, pady=5)
        class_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=class_var).pack(fill=tk.X, padx=10)

        def save():
            code = code_var.get().strip()
            name = name_var.get().strip()
            class_id = class_var.get().strip()
            if not code or not name:
                messagebox.showwarning("Aviso", "Codigo e nome sao obrigatorios")
                return
            conn = get_conn()
            try:
                conn.execute("INSERT INTO students (code, name, class_id) VALUES (?, ?, ?)",
                             (code, name, class_id))
                conn.commit()
                self.load_students()
                dialog.destroy()
            except sqlite3.IntegrityError:
                messagebox.showerror("Erro", "Codigo ja existe")
            finally:
                conn.close()

        ttk.Button(dialog, text="Salvar", command=save).pack(pady=10)

    def edit_student_dialog(self):
        selection = self.students_tree.selection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione um aluno")
            return

        item = self.students_tree.item(selection[0])
        values = item["values"]

        dialog = tk.Toplevel(self.root)
        dialog.title("Editar Aluno")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Codigo:").pack(anchor=tk.W, padx=10, pady=5)
        code_var = tk.StringVar(value=values[1])
        ttk.Entry(dialog, textvariable=code_var).pack(fill=tk.X, padx=10)

        ttk.Label(dialog, text="Nome:").pack(anchor=tk.W, padx=10, pady=5)
        name_var = tk.StringVar(value=values[2])
        ttk.Entry(dialog, textvariable=name_var).pack(fill=tk.X, padx=10)

        ttk.Label(dialog, text="Turma:").pack(anchor=tk.W, padx=10, pady=5)
        class_var = tk.StringVar(value=values[3])
        ttk.Entry(dialog, textvariable=class_var).pack(fill=tk.X, padx=10)

        def save():
            code = code_var.get().strip()
            name = name_var.get().strip()
            class_id = class_var.get().strip()
            if not code or not name:
                messagebox.showwarning("Aviso", "Codigo e nome sao obrigatorios")
                return
            conn = get_conn()
            try:
                conn.execute("UPDATE students SET code=?, name=?, class_id=? WHERE id=?",
                             (code, name, class_id, values[0]))
                conn.commit()
                self.load_students()
                dialog.destroy()
            except sqlite3.IntegrityError:
                messagebox.showerror("Erro", "Codigo ja existe")
            finally:
                conn.close()

        ttk.Button(dialog, text="Salvar", command=save).pack(pady=10)

    def delete_student(self):
        selection = self.students_tree.selection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione um aluno")
            return

        if not messagebox.askyesno("Confirmar", "Excluir aluno e todos seus dados?"):
            return

        item = self.students_tree.item(selection[0])
        student_id = item["values"][0]

        conn = get_conn()
        conn.execute("DELETE FROM embeddings WHERE student_id = ?", (student_id,))
        conn.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()
        conn.close()

        self.load_students()

    def register_face_dialog(self):
        selection = self.students_tree.selection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione um aluno")
            return

        item = self.students_tree.item(selection[0])
        values = item["values"]
        self.reg_code_var.set(values[1])
        self.notebook.select(0)  # Switch to camera tab

    # ===== ATTENDANCE METHODS =====
    def load_attendance(self):
        for item in self.att_tree.get_children():
            self.att_tree.delete(item)

        code = self.att_code_var.get().strip()
        start = self.att_start_var.get()
        end = self.att_end_var.get()

        conn = get_conn()
        query = """
            SELECT a.id, s.name, s.code, a.timestamp, a.type
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE 1=1
        """
        params = []
        if code:
            query += " AND s.code = ?"
            params.append(code)
        if start:
            query += " AND date(a.timestamp) >= ?"
            params.append(start)
        if end:
            query += " AND date(a.timestamp) <= ?"
            params.append(end)
        query += " ORDER BY a.timestamp DESC LIMIT 500"

        rows = conn.execute(query, params).fetchall()
        conn.close()

        for r in rows:
            dt = datetime.fromisoformat(r[3]).strftime("%d/%m/%Y %H:%M:%S")
            self.att_tree.insert("", tk.END, values=(r[0], r[1], r[2], dt, r[4]))

    def clear_attendance_filter(self):
        self.att_code_var.set("")
        self.att_start_var.set(datetime.now().strftime("%Y-%m-%d"))
        self.att_end_var.set(datetime.now().strftime("%Y-%m-%d"))
        self.load_attendance()

    # ===== REPORTS METHODS =====
    def generate_report(self):
        for item in self.report_tree.get_children():
            self.report_tree.delete(item)

        class_id = self.report_class_var.get().strip()
        date = self.report_date_var.get()

        conn = get_conn()
        rows = conn.execute("""
            SELECT s.code, s.name,
                   MIN(CASE WHEN a.type='entry' THEN a.timestamp END) as entry,
                   MAX(CASE WHEN a.type='exit' THEN a.timestamp END) as exit
            FROM students s
            LEFT JOIN attendance a ON s.id = a.student_id AND date(a.timestamp) = ?
            WHERE s.class_id = ? OR ? = ''
            GROUP BY s.id
        """, (date, class_id, class_id)).fetchall()
        conn.close()

        for r in rows:
            entry = r[2]
            exit_ = r[3]
            if entry:
                entry = datetime.fromisoformat(entry).strftime("%H:%M:%S")
            if exit_:
                exit_ = datetime.fromisoformat(exit_).strftime("%H:%M:%S")

            status = "Presente" if entry else "Ausente"
            if entry and exit_:
                status = "Completo"

            self.report_tree.insert("", tk.END, values=(r[0], r[1], entry or "-", exit_ or "-", status))

    def export_report_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return

        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Codigo", "Nome", "Entrada", "Saida", "Status"])
            for item in self.report_tree.get_children():
                writer.writerow(self.report_tree.item(item)["values"])
        messagebox.showinfo("Sucesso", f"Relatorio exportado para {path}")

    # ===== UTILS =====
    def log_event(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.log_text.see(tk.END)

    def on_close(self):
        self.stop_camera()
        self.stop_recognition_thread()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()