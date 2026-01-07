import os
import cv2
import base64
import torch
import psycopg2
import psycopg2.extras
import numpy as np
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1

# ===================== CONFIG =====================

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT", "5432"),
}

MATCH_THRESHOLD = 0.60
DEVICE = "cpu"   # Force CPU for Render stability

# ===================== MODELS =====================

mtcnn = MTCNN(
    keep_all=False,
    device=DEVICE,
    min_face_size=80
)

facenet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(DEVICE)

# ===================== DATABASE =====================

def connect_db():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    return conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

def setup_db(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            roll TEXT PRIMARY KEY,
            name TEXT,
            course TEXT,
            emb_left FLOAT8[],
            emb_center FLOAT8[],
            emb_right FLOAT8[]
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id SERIAL PRIMARY KEY,
            roll TEXT,
            name TEXT,
            course TEXT,
            time TIMESTAMP,
            confidence FLOAT8
        );
    """)

# ===================== UTILS =====================

def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def cosine_sim(a, b):
    return float(np.dot(a, b))

def get_embedding(face):
    face = cv2.resize(face, (160, 160))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    t = torch.tensor(face).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        emb = facenet(t).cpu().numpy()[0]
    return normalize(emb)

# ===================== IMAGE PROCESSING =====================

def process_web_image(base64_img):
    try:
        if "," in base64_img:
            base64_img = base64_img.split(",")[1]

        img_bytes = base64.b64decode(base64_img)
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box, _ = mtcnn.detect(rgb)

        if box is None:
            return None

        x1, y1, x2, y2 = map(int, box[0])
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            return None

        return get_embedding(face)

    except Exception:
        return None

# ===================== REGISTRATION =====================

def register_student_web(cur, roll, name, course, images):
    try:
        emb_center = process_web_image(images["center"])
        emb_left = process_web_image(images["left"])
        emb_right = process_web_image(images["right"])

        if not all([emb_center, emb_left, emb_right]):
            return {"status": "error", "message": "Face not detected in all images"}

        cur.execute("""
            INSERT INTO students (roll, name, course, emb_left, emb_center, emb_right)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (roll) DO UPDATE SET
                name = EXCLUDED.name,
                course = EXCLUDED.course,
                emb_left = EXCLUDED.emb_left,
                emb_center = EXCLUDED.emb_center,
                emb_right = EXCLUDED.emb_right
        """, (
            roll, name, course,
            emb_left.tolist(),
            emb_center.tolist(),
            emb_right.tolist()
        ))

        return {"status": "success"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# ===================== ATTENDANCE =====================

def load_students(cur):
    cur.execute("SELECT * FROM students")
    data = []
    for s in cur.fetchall():
        data.append((
            s,
            [
                normalize(s["emb_left"]),
                normalize(s["emb_center"]),
                normalize(s["emb_right"])
            ]
        ))
    return data

def mark_attendance(cur, embedding):
    students = load_students(cur)
    if not students:
        return {"status": "no_student"}

    best_score = -1
    best_student = None

    for s, embs in students:
        for db_emb in embs:
            score = cosine_sim(embedding, db_emb)
            if score > best_score:
                best_score = score
                best_student = s

    if best_score >= MATCH_THRESHOLD:
        ts = datetime.now()
        cur.execute("""
            INSERT INTO attendance (roll, name, course, time, confidence)
            VALUES (%s,%s,%s,%s,%s)
        """, (
            best_student["roll"],
            best_student["name"],
            best_student["course"],
            ts,
            best_score
        ))

        return {
            "status": "success",
            "roll": best_student["roll"],
            "name": best_student["name"],
            "confidence": float(best_score),
            "time": ts.isoformat()
        }

    return {"status": "unknown"}
