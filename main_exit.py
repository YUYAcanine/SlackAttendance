
import cv2
import numpy as np
import os

from insightface.app import FaceAnalysis
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import datetime
import pyttsx3

import json

#GAS用
import requests
GAS_URL = "https://script.google.com/macros/s/AKfycby1-BKUwUAgA4nA-CuLZnQ6aQWnGGzB_xg0qc1-jwq0wPt0u7gj_2FhTw-TVVTRg4mz/exec"
from dotenv import load_dotenv
load_dotenv()
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")

#GAS削除用（出口カメラ）
def send_exit(name):
    data = {
        "event": "exit",
        "name": name
    }
    res = requests.post(
        GAS_URL,
        data=json.dumps(data),
        headers={"Content-Type": "application/json"}
    )
    #doPostの返り血取得
    result = res.json()
    print(result)

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# 全員分のベクトルを読み込む
EMBEDDING_DIR = "embeddings"
known_faces = {}
name_list = {}
engine=pyttsx3.init()
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)
engine.setProperty('rate',150)
kidoubi=datetime.date.today()

for filename in os.listdir(EMBEDDING_DIR):
    if filename.endswith(".npy"):
        name = os.path.splitext(filename)[0]
        emb = np.load(os.path.join(EMBEDDING_DIR, filename))
        known_faces[name] = emb

cap = cv2.VideoCapture(0) #1→外部カメラ、0→内臓カメラ

def preprocess(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return frame


def gamma_correction(img, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def brighten(img, alpha=1.3, beta=40):
    # alpha: コントラスト, beta: 明るさ
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    #frame = preprocess(frame)
    #frame = gamma_correction(frame, 1.5)
    #frame = brighten(frame, 1.3, 40)
    faces = app.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        face_img = frame[y1:y2, x1:x2]

        # ===== 顔補正（可変強度）=====
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)

        if mean_val < 60:
            # かなり暗い
            face_img = gamma_correction(face_img, 1.8)
            face_img = brighten(face_img, 1.4, 70)

        elif mean_val < 90:
            # 少し暗い
            face_img = gamma_correction(face_img, 1.4)
            face_img = brighten(face_img, 1.2, 30)

        # 明るいときは何もしない

        # 白飛び防止
        face_img = np.clip(face_img, 0, 220)

        frame[y1:y2, x1:x2] = face_img

        # ===== embedding =====
        emb = face.embedding


        best_match_exit = "Unknown"
        best_sim = 0

        for name, known_emb in known_faces.items():
            sim = np.dot(emb, known_emb) / (np.linalg.norm(emb) * np.linalg.norm(known_emb))
            if sim > best_sim:
                best_sim = sim
                best_match_exit = name
                    
        label = f"{best_match_exit} ({best_sim:.2f})" if best_sim > 0.1 else "Unknown"


        
        if best_sim < 0.3:
            best_match_exit = "Unknown"
        
        if best_match_exit in name_list:
            send_exit(best_match_exit) # GAS送信

            if name_list[best_match_exit] != datetime.date.today():
                passed_daytime = datetime.date.today() - name_list[best_match_exit]
                passed_days = passed_daytime.days

                name_list[best_match_exit] = datetime.date.today()

        else:
            if best_match_exit != "Unknown": #and kidoubi!=datetime.date.today():
                name_list[best_match_exit] = datetime.date.today()
                send_exit(best_match_exit) #GAS送信

        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
