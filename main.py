
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis


from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import datetime

from dotenv import load_dotenv
load_dotenv()
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")

def SendToSlackMessage(message):
    client = WebClient(token=SLACK_BOT_TOKEN) 
    
    name_map = {"yuya":"川辺",
                "yusei":"行平",
                "satoshi":"稲垣",
                "hane":"羽根",
                "handa":"半田",
                "hashimoto":"橋本",
                "izumitani":"泉谷",
                "kuribayashi":"栗林",
                "matsumoto":"松元",
                "nishida":"西田",
                "nomura":"野村",
                "noto":"能登",
                "nozaki":"能崎",
                "ono":"大野",
                "sano":"佐野",
                "tanaka":"田中",
                "tokutomi":"徳富",
                "kishimura":"岸村"
                }
    name_map_read = {"yuya":"かわべ",
                    "yusei":"ゆきひら",
                    "satoshi":"いながき",
                    "hane":"はね",
                    "handa":"はんだ",
                    "hashimoto":"はしもと",
                    "izumitani":"いずみたに",
                    "kuribayashi":"くりばやし",
                    "matsumoto":"まつもと",
                    "nishida":"にしだ",
                    "nomura":"のむら",
                    "noto":"のと",
                    "nozaki":"のざき",
                    "ono":"おおの",
                    "sano":"さの",
                    "tanaka":"たなか",
                    "tokutomi":"とくとみ",
                    "kishimura":"きしむら"
                    }

    response=client.chat_postMessage(channel='010_lab-in', text = name_map[message] + "出校しました")
    engine.say(name_map_read[message]+"さん、おはようございます")
    engine.runAndWait()



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
                    
        label = f"{best_match} ({best_sim:.2f})" if best_sim > 0.5 else "Unknown"


        
        if best_sim < 0.5:
            best_match = "Unknown"


        
        if best_match in name_list:
            if name_list[best_match] != datetime.date.today():
                SendToSlackMessage(best_match)
                name_list[best_match] = datetime.date.today()

        else:
            if best_match != "Unknown":
                name_list[best_match] = datetime.date.today()
                SendToSlackMessage(best_match)


        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
