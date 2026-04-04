
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
GAS_URL = "https://script.google.com/macros/s/AKfycbxgHr0v6UoXf0Nbc0O-ihQj87qlkf9w97hzq8QCpkKGsfA-zI62ABvtJ9PAIvXs5og2Ew/exec"
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

cap = cv2.VideoCapture(1) #1→外部カメラ、0→内臓カメラ

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        emb = face.embedding
        best_match_exit = "Unknown"
        best_sim = 0

        for name, known_emb in known_faces.items():
            sim = np.dot(emb, known_emb) / (np.linalg.norm(emb) * np.linalg.norm(known_emb))
            if sim > best_sim:
                best_sim = sim
                best_match_exit = name
                    
        label = f"{best_match_exit} ({best_sim:.2f})" if best_sim > 0.1 else "Unknown"


        
        if best_sim < 0.25:
            best_match_exit = "Unknown"


        
        if best_match_exit in name_list:
            send_exit(best_match_exit) #GAS送信
            if name_list[best_match_exit] != datetime.date.today():
                passed_daytime=name_list[best_match_exit]- datetime.date.today()
                passed_days=passed_daytime.day
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
