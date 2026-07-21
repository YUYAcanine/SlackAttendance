# SlackProject: 顔認識システム

このプロジェクトは InsightFace を使ったリアルタイム顔認識システム
python3.10.9を使用

ルートディレクトリに作成
.env SLACK_BOT_TOKEN = "xoxb-qwertyuiop"

1. リポジトリをクローン
2. 仮想環境を作成
3. ライブラリをインストール

SlackProject/known_faces/name/1,2,3,4,,,.jpg
extract_embeddings.pyで顔特徴のベクトルを生成
SlackProject/embenddings/---.npy作成

`main.py`を実行すると、次の3つがまとめて起動します。

- 入口カメラ: `main_entry.py`
- 出口カメラ: `main_exit.py`
- 在室状況Webサーバー: `http://127.0.0.1:8000`

```powershell
python main.py
```

同じLAN内の別端末から表示する場合は、実行PCのIPアドレスを使って
`http://PCのIPアドレス:8000`へアクセスします。終了するときは、ターミナルで
`Ctrl+C`を押すと入口・出口カメラも一緒に終了します。

在室状態は`attendance_state.json`へ保存され、日付が変わると自動的にリセットされます。
LEDサーバーURLを変更する場合は、環境変数`LED_SERVER_URL`へ指定できます。
