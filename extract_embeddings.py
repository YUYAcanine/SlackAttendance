import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

KNOWN_DIR = "known_faces"
EMBEDDING_DIR = "embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

for person_name in os.listdir(KNOWN_DIR):
    person_path = os.path.join(KNOWN_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    embeddings = []
    for filename in os.listdir(person_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(person_path, filename)
        img = cv2.imread(img_path)
        faces = app.get(img)
        if faces:
            embeddings.append(faces[0].embedding)
            print(f"[{person_name}] {filename} -> OK")
        else:
            print(f"[{person_name}] {filename} -> 顔が見つかりませんでした")

    if embeddings:
        avg_emb = np.mean(embeddings, axis=0)
        np.save(os.path.join(EMBEDDING_DIR, f"{person_name}.npy"), avg_emb)
        print(f"{person_name} のベクトルを保存しました\n")
    else:
        print(f"{person_name} の顔ベクトルが作成できませんでした\n")
