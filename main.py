
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
                "hane":"東吾"
                }

    response=client.chat_postMessage(channel='work', text = name_map[message] + "出校しました")


app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# 全員分のベクトルを読み込む
EMBEDDING_DIR = "embeddings"
known_faces = {}
name_list = {}


for filename in os.listdir(EMBEDDING_DIR):
    if filename.endswith(".npy"):
        name = os.path.splitext(filename)[0]
        emb = np.load(os.path.join(EMBEDDING_DIR, filename))
        known_faces[name] = emb

cap = cv2.VideoCapture(0)

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
            else:
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
