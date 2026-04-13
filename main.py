
import cv2
import numpy as np
import os

from insightface.app import FaceAnalysis
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import datetime
import pyttsx3

#GAS用
import requests
import json

GAS_URL = "https://script.google.com/macros/s/AKfycby1-BKUwUAgA4nA-CuLZnQ6aQWnGGzB_xg0qc1-jwq0wPt0u7gj_2FhTw-TVVTRg4mz/exec"

from dotenv import load_dotenv
load_dotenv()
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")

#リアルタイムGAS表示用（入口カメラ）
def send_entry(name):
    data = {
        "event": "entry",
        "name": name
    }

    #戻り値の定義とdopostの実行
    res = requests.post(
        GAS_URL,
        data=json.dumps(data),
        headers={"Content-Type": "application/json"}
    )

    #doPostの返り血取得
    result = res.json()
    print(result)

def speak(text, speaker=8):  
    query = requests.post(
        "http://127.0.0.1:50021/audio_query",
        params={"text": text, "speaker": speaker}
    ).json()

    # ===== 調整 =====
    query["volumeScale"] = 4.0
    query["speedScale"] = 1.0
    query["intonationScale"] = 1.0

    voice = requests.post(
        "http://127.0.0.1:50021/synthesis",
        params={"speaker": speaker},
        data=json.dumps(query)
    )

    with open("voice.wav", "wb") as f:
        f.write(voice.content)

    os.system("start voice.wav")


# =========================
# Slack + 音声
# =========================
def SendToSlackMessage(message, passed_days):
    client = WebClient(token=SLACK_BOT_TOKEN)

    name_map = {
        "yuya":"川辺",
        "yusei":"行平",
        "satoshi":"稲垣",
        "hane":"羽根",
        "hashimoto":"橋本",
        "kuribayashi":"栗林",
        "matsumoto":"松元",
        "nishida":"西田",
        "nomura":"野村",
        "ono":"大野",
        "sano":"佐野",
        "tanaka":"田中",
        "tokutomi":"徳富",
        "yoshida":"吉田",
        "kondo":"近藤" ,
        "hasegawa":"長谷川",
        "hoashi":"帆足",
        "honda":"本田",
        "hujiwara":"藤原",
        "kamigiri":"上桐",
        "shibata":"柴田",
        "tomioka":"富岡",
        "katsuyama":"勝山",
    }

    name_map_read = {
        "yuya":"かわべ",
        "yusei":"ゆきひら",
        "satoshi":"いながき",
        "hane":"はね",
        "hashimoto":"はしもと",
        "kuribayashi":"くりばやし",
        "matsumoto":"まつもと",
        "nishida":"にしだ",
        "nomura":"のむら",
        "ono":"おおの",
        "sano":"さの",
        "tanaka":"たなか",
        "tokutomi":"とくとみ",
        "yoshida":"よしだ",
        "kondo":"こんどう",
        "hasegawa":"はせがわ",
        "hoashi":"ほあし",
        "honda":"ほんだ",
        "hujiwara":"ふじわら",
        "kamigiri":"かみぎり",
        "shibata":"しばた",
        "tomioka":"とみおか",
        "katsuyama":"かつやま",
    }

    # Slack投稿
    client.chat_postMessage(
        channel='010_lab-in',
        text=name_map[message] + "出校しました"
    )

    # 音声分岐
    name = name_map_read[message]
    hour = datetime.datetime.now().hour

    if passed_days > 3:
        speak(f"{name}さん、おひさしぶりです")

    else:
        if hour < 4:
            speak("通報しました")

        elif hour < 12:
            speak(f"{name}さん、おはようございます")

        elif hour < 18:
            speak(f"{name}さん、こんにちは")

        else:
            speak(f"{name}さん、こんばんは")


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
        best_match = "Unknown"
        best_sim = 0

        for name, known_emb in known_faces.items():
            sim = np.dot(emb, known_emb) / (np.linalg.norm(emb) * np.linalg.norm(known_emb))
            if sim > best_sim:
                best_sim = sim
                best_match = name
                    
        label = f"{best_match} ({best_sim:.2f})" if best_sim > 0.1 else "Unknown"


        
        if best_sim < 0.35:
            best_match = "Unknown"
        
        if best_match in name_list:
            
            send_entry(best_match) #GAS送信

            if name_list[best_match] != datetime.date.today():
                passed_daytime=name_list[best_match]- datetime.date.today()
                passed_days=passed_daytime.days
                SendToSlackMessage(best_match,passed_days)
                
                name_list[best_match] = datetime.date.today()

        else:
            if best_match != "Unknown": #and kidoubi!=datetime.date.today():
                name_list[best_match] = datetime.date.today()
                SendToSlackMessage(best_match,0)
                send_entry(best_match) #GAS送信

        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()