import os
import cv2
import time
import math
import sqlite3
import csv
import numpy as np
import jwt
import datetime
import smtplib
from email.mime.text import MIMEText
from functools import wraps
from collections import deque
from flask import Flask, render_template, request, redirect, Response, jsonify, send_file
from ultralytics import YOLO
from werkzeug.security import generate_password_hash, check_password_hash

# ===================== APP CONFIG =====================
app = Flask(__name__)
app.secret_key = "super_secret_key"

UPLOAD_FOLDER = "videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB = "users.db"
JWT_SECRET = "crowd_secure_key"
JWT_ALGO = "HS256"

model = YOLO("yolov8n.pt")

FRAME_SKIP = 3
CONF = 0.35

# ===================== EMAIL ALERT CONFIG =====================
EMAIL_SENDER = "abhay.s3.prakash@gmail.com"
EMAIL_PASSWORD = "supfmvlnouyqwjuc"
EMAIL_RECEIVER = "abhayprakashjob101@gmail.com"

ALERT_THRESHOLD = 20
email_sent = False

# ===================== GLOBAL STATE =====================
video_path = None
processing_done = False

stats = {
    "live": 0,
    "total": 0,
    "average": 0,
    "duration": 0,
    "size": 0,
    "zone_count": 0,
    "status": "LOW"
}

people_history = deque(maxlen=300)
unique_track_ids = set()

heatmap = None
HEATMAP_DECAY = 0.95

ZONE = {"x1": 200, "y1": 100, "x2": 500, "y2": 400}

# ===================== DATABASE =====================
def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY,
            video TEXT,
            total INTEGER,
            average REAL,
            max_count INTEGER,
            duration INTEGER,
            date TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ===================== JWT =====================
def create_token(username):
    return jwt.encode({
        "user": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    }, JWT_SECRET, algorithm=JWT_ALGO)

def jwt_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.cookies.get("token")
        if not token:
            return redirect("/")
        try:
            jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        except:
            return redirect("/")
        return f(*args, **kwargs)
    return wrapper

# ===================== AUTH =====================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username=?", (u,))
        row = cur.fetchone()
        conn.close()

        if row and check_password_hash(row[0], p):
            token = create_token(u)
            resp = redirect("/dashboard")
            resp.set_cookie("token", token, httponly=True, samesite="Strict")
            return resp

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        if len(p) < 6:
            return render_template("register.html", error="Password too short")

        try:
            conn = sqlite3.connect(DB)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users(username,password) VALUES(?,?)",
                (u, generate_password_hash(p))
            )
            conn.commit()
            conn.close()
        except sqlite3.IntegrityError:
            return render_template("register.html", error="User exists")

        return redirect("/")

    return render_template("register.html")

# ===================== DASHBOARD =====================
@app.route("/dashboard", methods=["GET", "POST"])
@jwt_required
def dashboard():
    global video_path, heatmap, processing_done, email_sent

    if request.method == "POST":
        file = request.files["video"]

        video_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(video_path)

        unique_track_ids.clear()
        people_history.clear()
        heatmap = None
        processing_done = False
        email_sent = False

        return redirect("/stream")

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT * FROM history ORDER BY id DESC")
    history = cur.fetchall()
    conn.close()

    return render_template("dashboard.html", history=history, threshold=ALERT_THRESHOLD)

# ===================== ALERT THRESHOLD =====================
@app.route("/set_threshold", methods=["POST"])
@jwt_required
def set_threshold():
    global ALERT_THRESHOLD
    ALERT_THRESHOLD = int(request.form["threshold"])
    return redirect("/dashboard")

# ===================== EMAIL ALERT =====================
def send_email_alert(live):
    msg = MIMEText(f"""
🚨 SafeCrowd AI ALERT 🚨

Overcrowding detected!

Live Crowd Count: {live}
Threshold Limit: {ALERT_THRESHOLD}
""")

    msg["Subject"] = "🚨 SafeCrowd AI – Crowd Alert"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print("Email error:", e)

# ===================== VIDEO PROCESS =====================
def generate_frames():
    global heatmap, email_sent, processing_done

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stats["duration"] = int(frames / fps)
    stats["size"] = round(os.path.getsize(video_path) / (1024 * 1024), 2)

    frame_no = 0
    max_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % FRAME_SKIP != 0:
            continue

        if heatmap is None:
            heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

        small = cv2.resize(frame, (640, 360))
        results = model.track(
            small,
            persist=True,
            conf=CONF,
            iou=0.5,
            tracker="bytetrack.yaml"
        )[0]

        sx, sy = frame.shape[1] / 640, frame.shape[0] / 360
        live = 0
        zone_count = 0

        if results.boxes:
            for box in results.boxes:

                if int(box.cls[0]) != 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                cx, cy = (x1+x2)//2, (y1+y2)//2

                live += 1

                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is not None:
                    unique_track_ids.add(track_id)

                if ZONE["x1"] < cx < ZONE["x2"] and ZONE["y1"] < cy < ZONE["y2"]:
                    zone_count += 1

                heatmap[cy, cx] += 1

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                if track_id is not None:
                    cv2.putText(
                        frame,
                        f"ID {track_id}",
                        (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,255),
                        2
                    )

        cv2.rectangle(frame, (ZONE["x1"],ZONE["y1"]), (ZONE["x2"],ZONE["y2"]), (255,0,0), 2)

        people_history.append(live)
        max_count = max(max_count, live)

        stats["live"] = live
        stats["zone_count"] = zone_count
        stats["total"] = len(unique_track_ids)
        stats["average"] = round(sum(people_history) / len(people_history), 2)
        stats["status"] = "LOW" if live < 10 else "MEDIUM" if live < ALERT_THRESHOLD else "HIGH"

        if live >= ALERT_THRESHOLD and not email_sent:
            send_email_alert(live)
            email_sent = True
        if live < ALERT_THRESHOLD:
            email_sent = False

        heatmap *= HEATMAP_DECAY
        if heatmap.max() > 0:
            hm = cv2.applyColorMap(
                np.uint8(255 * (heatmap / heatmap.max())),
                cv2.COLORMAP_JET
            )
            frame = cv2.addWeighted(frame, 0.6, hm, 0.4, 0)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

        time.sleep(0.03)

    cap.release()
    processing_done = True

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO history(video,total,average,max_count,duration,date)
        VALUES(?,?,?,?,?,?)
    """, (
        os.path.basename(video_path),
        stats["total"],
        stats["average"],
        max_count,
        stats["duration"],
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    ))
    conn.commit()
    conn.close()

# ===================== ROUTES =====================
@app.route("/stream")
@jwt_required
def stream():
    return render_template("stream.html")

@app.route("/video_feed")
@jwt_required
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/stats")
@jwt_required
def api_stats():
    return jsonify({**stats, "done": processing_done})

@app.route("/download_report")
@jwt_required
def download_report():
    csv_file = "report.csv"
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT * FROM history")
    rows = cur.fetchall()
    conn.close()

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID","Video","Total","Average","Max","Duration","Date"])
        writer.writerows(rows)

    return send_file(csv_file, as_attachment=True)

@app.route("/logout")
def logout():
    resp = redirect("/")
    resp.delete_cookie("token")
    return resp

if __name__ == "__main__":
    app.run(debug=True)
