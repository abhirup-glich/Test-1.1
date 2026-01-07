from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
import time
import threading
import sys
import base64
from datetime import datetime
import psycopg2
import psycopg2.extras
from facenet_pytorch import MTCNN, InceptionResnetV1
from tabulate import tabulate
import argparse
import os
app = Flask(__name__)
# ===================== CONFIG =====================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "face_recognition_db"),
    "user": os.getenv("DB_USER", "postgres"),
    # FIX 3: Removed hardcoded default password for security
    "password": os.getenv("DB_PASSWORD", ""), 
    "port": os.getenv("DB_PORT", "5432"),
}

MATCH_THRESHOLD = 0.65
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

exit_attendance = False

# ===================== MODELS =====================
mtcnn = MTCNN(keep_all=False, device=DEVICE)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
# This path relies on 'shape_predictor_68_face_landmarks.dat' being in the same directory as logic.py
# ===================== DATABASE =====================

def connect_db():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    return conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

def setup_db(cur):
    # Create tables only if they don't exist
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
        
        CREATE TABLE IF NOT EXISTS classes (
            id SERIAL PRIMARY KEY,
            course TEXT,
            start_time TIMESTAMP DEFAULT NOW(),
            notes TEXT
        );
        
        -- Add comments to document the schema (only if they don't exist)
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_description 
                WHERE objoid = 'students'::regclass AND objsubid = 0
            ) THEN
                COMMENT ON TABLE students IS 'Stores student information and face embeddings';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM pg_description 
                WHERE objoid = 'attendance'::regclass AND objsubid = 0
            ) THEN
                COMMENT ON TABLE attendance IS 'Stores attendance records';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM pg_description 
                WHERE objoid = 'classes'::regclass AND objsubid = 0
            ) THEN
                COMMENT ON TABLE classes IS 'Logs each class session start';
            END IF;
        END $$;
    """)

# ===================== UTILS =====================

def normalize(v):
    v = np.asarray(v, dtype=np.float32).flatten()
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def cosine_sim(a, b):
    a, b = normalize(a), normalize(b)
    if a.shape != b.shape:
        return -1.0
    return float(np.dot(a, b))

def get_embedding(face):
    face = cv2.resize(face, (160,160))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    t = torch.tensor(face).permute(2,0,1).float().unsqueeze(0)/255.0
    t = t.to(DEVICE)
    with torch.no_grad():
        emb = facenet(t).cpu().numpy()[0]
    return normalize(emb)

# ===================== REGISTRATION =====================

def capture_embeddings(cap, prompt, samples=20):
    data = []
    while len(data) < samples:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box, _ = mtcnn.detect(rgb)
        if box is not None:
            x1,y1,x2,y2 = map(int, box[0])
            face = frame[y1:y2, x1:x2]
            data.append(get_embedding(face))
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"{prompt} {len(data)}/{samples}",
                    (30,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)
        cv2.imshow("Register",frame)
        cv2.waitKey(1)
    return normalize(np.mean(data, axis=0))

def register_student(cur):
    roll = input("Roll: ")
    name = input("Name: ")
    course = input("Course: ")

    cap = cv2.VideoCapture(0)
    center = capture_embeddings(cap, "LOOK STRAIGHT")
    left = capture_embeddings(cap, "TURN LEFT")
    right = capture_embeddings(cap, "TURN RIGHT")
    cap.release()
    cv2.destroyAllWindows()

    cur.execute("""
        INSERT INTO students (roll, name, course, emb_left, emb_center, emb_right) 
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (roll) DO UPDATE SET
        name = EXCLUDED.name,
        course = EXCLUDED.course,
        emb_left = EXCLUDED.emb_left,
        emb_center = EXCLUDED.emb_center,
        emb_right = EXCLUDED.emb_right
    """,(roll,name,course,
         left.tolist(),center.tolist(),right.tolist()))

    print(f"Registered: {name}")

def process_web_image(base64_str):
    try:
        # Decode base64
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect face
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box, _ = mtcnn.detect(rgb)
        
        if box is None:
            return None
            
        # Get largest face if multiple
        # Simple approach: just take first one for now or max area
        x1,y1,x2,y2 = map(int, box[0])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
            
        return get_embedding(face)
    except Exception as e:
        print(f"Error processing web image: {e}")
        return None

def register_student_web(cur, roll, name, course, images):
    # images is dict with keys: center, left, right
    
    emb_center = process_web_image(images['center'])
    if emb_center is None:
        return {'status': 'error', 'message': 'Face not detected in Center photo'}
        
    emb_left = process_web_image(images['left'])
    if emb_left is None:
        return {'status': 'error', 'message': 'Face not detected in Left photo'}
        
    emb_right = process_web_image(images['right'])
    if emb_right is None:
        return {'status': 'error', 'message': 'Face not detected in Right photo'}
        
    try:
        cur.execute("""
            INSERT INTO students (roll, name, course, emb_left, emb_center, emb_right) 
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (roll) DO UPDATE SET
            name = EXCLUDED.name,
            course = EXCLUDED.course,
            emb_left = EXCLUDED.emb_left,
            emb_center = EXCLUDED.emb_center,
            emb_right = EXCLUDED.emb_right
        """,(roll, name, course,
             emb_left.tolist(), emb_center.tolist(), emb_right.tolist()))
             
        return {'status': 'success', 'message': f'Student {name} registered successfully'}
    except Exception as e:
        print(f"DB Error: {e}")
        return {'status': 'error', 'message': str(e)}

# ===================== ATTENDANCE =====================

def load_students(cur):
    cur.execute("SELECT * FROM students")
    students = []
    for s in cur.fetchall():
        students.append((
            s,
            [normalize(s["emb_left"]),
             normalize(s["emb_center"]),
             normalize(s["emb_right"])]
        ))
    return students

def terminal_listener():
    global exit_attendance
    while True:
        if sys.stdin.readline().strip().lower() == "q":
            exit_attendance = True
            break
            
def record_class_start(cur, course=None, notes=None):
    try:
        cur.execute("""
            INSERT INTO classes (course, notes)
            VALUES (%s, %s)
            RETURNING id, start_time, course
        """, (course, notes))
        rec = cur.fetchone()
        return {
            "status": "success",
            "class_id": rec["id"],
            "start_time": rec["start_time"].isoformat(),
            "course": rec["course"]
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def mark_attendance(cur, video_path=None):
    cap = None
    try:
        students = load_students(cur)
        if not students:
            return {"error": "No students registered"}

        # Use video file if provided, otherwise use webcam
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)
            
        if not cap.isOpened():
            return {"error": "Could not open video source"}

        marked = set()
        start_time = time.time()
        attendance_marked = False
        saw_face = False
        result = {"status": "no_face", "name": None, "roll": None, "time": None, "confidence": 0.0}

        print(f"\nAttendance check starting - Source: {'File' if video_path else 'Webcam'}")

        while True:  # Process until end of stream or timeout
            # For webcam, limit to 5 seconds. For file, process until end or success
            if not video_path and time.time() - start_time > 5:
                break
                
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            box, _ = mtcnn.detect(rgb)

            if box is not None:
                x1, y1, x2, y2 = map(int, box[0])
                # Ensure coordinates are within frame
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                face = frame[y1:y2, x1:x2]
                
                # Skip if face is too small
                if face.size == 0:
                    continue
                
                saw_face = True
                

                if not video_path:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                can_mark = False
              # Liveness already validated on client (browser)
                can_mark = True

                        
                if can_mark and not attendance_marked:
                    emb = get_embedding(face)
                    best_score, best_student = -1, None

                    for s, embs in students:
                        if s["roll"] in marked:
                            continue
                        for db_emb in embs:
                            score = cosine_sim(emb, db_emb)
                            if score > best_score:
                                best_score, best_student = score, s

                    if best_score > 0.6:  # Confidence threshold
                        timestamp = datetime.now()
                        cur.execute("""
                            INSERT INTO attendance (roll, name, course, time, confidence)
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING id, time
                        """, (best_student["roll"], best_student["name"], 
                             best_student["course"], timestamp, best_score))
                        
                        # Get the inserted record
                        attendance_record = cur.fetchone()
                        marked.add(best_student["roll"])
                        print(f"Marked: {best_student['name']} ({best_score:.2f})")
                        attendance_marked = True
                        
                        # Update result
                        result = {
                            "status": "success",
                            "name": best_student["name"],
                            "roll": best_student["roll"],
                            "time": attendance_record["time"].isoformat(),
                            "confidence": float(best_score)
                        }
                        
                        if not video_path:
                            cv2.putText(frame, "ATTENDANCE MARKED", (40, 160),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            # If processing a file and we found a match, we can stop
                            break
                    else:
                        if not video_path:
                            cv2.putText(frame, "UNKNOWN FACE", (40, 160),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show remaining time only for webcam
            if not video_path:
                elapsed = time.time() - start_time
                remaining = max(0, 5 - int(elapsed))
                cv2.putText(frame, f"Time: {remaining}s", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Attendance", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        if not attendance_marked:
            result["status"] = "no_student" if saw_face else "no_face"
            result["time"] = datetime.now().isoformat()
            result["confidence"] = 0.0
            
    except Exception as e:
        print(f"Error in mark_attendance: {e}")
        return {"error": str(e)}
    finally:
        if cap and cap.isOpened():
            cap.release()
        if not video_path:
            cv2.destroyAllWindows()
    
    return result
# ===================== EDIT DATABASE =====================

def edit_database(cur):
    cur.execute("SELECT roll,name,course FROM students ORDER BY roll")
    rows = cur.fetchall()
    print(tabulate([[i+1,r["roll"],r["name"],r["course"]]
                    for i,r in enumerate(rows)],
                   headers=["SL","Roll","Name","Course"],
                   tablefmt="grid"))
    roll = input("Roll to edit: ")
    print("1.Update Name\n2.Update Course\n3.Delete")
    ch = input("Choice: ")

    if ch == "1":
        cur.execute("UPDATE students SET name=%s WHERE roll=%s",
                    (input("New Name: "),roll))
    elif ch == "2":
        cur.execute("UPDATE students SET course=%s WHERE roll=%s",
                    (input("New Course: "),roll))
    elif ch == "3":
        cur.execute("DELETE FROM students WHERE roll=%s",(roll,))
    print("Updated")

# ===================== DATABASE OPERATIONS =====================

def delete_last_attendance(cur):
    try:
        cur.execute("""
            DELETE FROM attendance 
            WHERE id = (
                SELECT id FROM attendance 
                ORDER BY time DESC 
                LIMIT 1
            )
            RETURNING id, name, time
        """)
        deleted = cur.fetchone()
        return deleted
    except Exception as e:
        print(f"Error deleting last attendance: {e}")
        return None

def clear_all_attendance(cur):
    try:
        cur.execute("DELETE FROM attendance")
        return True
    except Exception as e:
        print(f"Error clearing all attendance: {e}")
        return False

def fetch_all_students(cur):
    try:
        cur.execute("SELECT roll, name, course FROM students ORDER BY roll")
        return cur.fetchall()
    except Exception as e:
        print(f"Error fetching students: {e}")
        return []

def update_student_record(cur, roll, name=None, course=None):
    try:
        if name and course:
            cur.execute("UPDATE students SET name=%s, course=%s WHERE roll=%s", (name, course, roll))
        elif name:
            cur.execute("UPDATE students SET name=%s WHERE roll=%s", (name, roll))
        elif course:
            cur.execute("UPDATE students SET course=%s WHERE roll=%s", (course, roll))
        return True
    except Exception as e:
        print(f"Error updating student: {e}")
        return False

def delete_student_record(cur, roll):
    try:
        cur.execute("DELETE FROM students WHERE roll=%s", (roll,))
        return True
    except Exception as e:
        print(f"Error deleting student: {e}")
        return False

# ===================== MAIN =====================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Recognition Attendance System')
    parser.add_argument('--mode', choices=['register', 'attendance', 'view'], 
                       help='Run in specific mode: register, attendance, or view attendance')
    args = parser.parse_args()
    
    conn, cur = connect_db()
    setup_db(cur) # Setup DB tables before running modes
    
    if args.mode == 'register':
        register_student(cur)
        return
    elif args.mode == 'attendance':
        mark_attendance(cur)
        return
    elif args.mode == 'view':
        cur.execute("SELECT * FROM attendance ORDER BY time DESC")
        print(tabulate(cur.fetchall(),headers="keys",tablefmt="grid"))
        return
    
    while True:
        print("\n1.Register\n2.Attendance\n3.View Attendance\n4.Edit Database\n5.Exit")
        ch = input("Select: ")

        if ch=="1": register_student(cur)
        elif ch=="2": mark_attendance(cur)
        elif ch=="3":
            cur.execute("SELECT * FROM attendance ORDER BY time DESC")
            print(tabulate(cur.fetchall(),headers="keys",tablefmt="grid"))
        elif ch=="4": edit_database(cur)
        elif ch=="5": break

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/register", methods=["POST"])
def register_api():
    data = request.json
    conn, cur = connect_db()
    res = register_student_web(
        cur,
        data["roll"],
        data["name"],
        data["course"],
        data["images"]
    )
    cur.close()
    conn.close()
    return jsonify(res)

@app.route("/attendance", methods=["POST"])
def attendance_api():
    data = request.json
    emb = process_web_image(data["image"])
    if emb is None:
        return jsonify({"status": "no_face"})

    conn, cur = connect_db()
    students = load_students(cur)

    best_score, best_student = -1, None
    for s, embs in students:
        for db_emb in embs:
            score = cosine_sim(emb, db_emb)
            if score > best_score:
                best_score, best_student = score, s

    if best_score > 0.6:
        ts = datetime.now()
        cur.execute(
            "INSERT INTO attendance (roll,name,course,time,confidence) VALUES (%s,%s,%s,%s,%s)",
            (best_student["roll"], best_student["name"], best_student["course"], ts, best_score)
        )
        cur.close()
        conn.close()
        return jsonify({
            "status": "success",
            "name": best_student["name"],
            "roll": best_student["roll"],
            "confidence": best_score,
            "time": ts.isoformat()
        })

    cur.close()
    conn.close()
    return jsonify({"status": "unknown"})



