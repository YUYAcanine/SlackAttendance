import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError
from urllib.request import Request, urlopen


HOST = os.environ.get("STATUS_SERVER_HOST", "0.0.0.0")
PORT = int(os.environ.get("STATUS_SERVER_PORT", "8000"))
LED_SERVER = os.environ.get(
    "LED_SERVER_URL",
    "https://utilize-ignition-amber.ngrok-free.dev",
).rstrip("/")
PROJECT_DIR = Path(__file__).resolve().parent
STATE_FILE = PROJECT_DIR / "attendance_state.json"
STATE_LOCK = threading.Lock()

NAME_MAP = {
    "yuya": "川辺", "yusei": "行平", "satoshi": "稲垣", "hane": "羽根",
    "hashimoto": "橋本", "kuribayashi": "栗林", "matsumoto": "松元",
    "nishida": "西田", "nomura": "野村", "ono": "大野", "sano": "佐野",
    "tanaka": "田中", "tokutomi": "徳富", "yoshida": "吉田", "kondo": "近藤",
    "hasegawa": "長谷川", "hoashi": "帆足", "honda": "本田",
    "hujiwara": "藤原", "kamigiri": "上桐", "shibata": "柴田",
    "tomioka": "富岡", "katsuyama": "勝山", "yamada": "山田",
    "philip": "フィリップ",
}

PAGE = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>リアルタイム在室状況</title>
  <style>
    :root { font-family: system-ui, sans-serif; color: #172033; background: #f4f7fb; }
    body { max-width: 1200px; margin: 0 auto; padding: 28px; }
    h1 { margin-bottom: 4px; }
    #summary { color: #5b6474; margin-bottom: 24px; }
    #people { display: flex; flex-wrap: wrap; gap: 16px; }
    .person { min-width: 150px; padding: 22px 28px; border-radius: 16px;
      background: white; box-shadow: 0 5px 20px #20305018; font-size: 28px;
      font-weight: 700; text-align: center; border-left: 8px solid #22a06b; }
    .person.out { color: #7b8493; background: #e8ebf0; border-color: #9aa2ae; }
    .status { margin-top: 7px; font-size: 13px; font-weight: 500; }
    #updated { margin-top: 28px; color: #778090; }
  </style>
</head>
<body>
  <h1>リアルタイム在室状況</h1>
  <div id="summary">読み込み中...</div>
  <main id="people"></main>
  <div id="updated"></div>
  <script>
    const names = __NAME_MAP__;
    function escapeHtml(value) {
      const node = document.createElement('div');
      node.textContent = value;
      return node.innerHTML;
    }
    async function refresh() {
      try {
        const response = await fetch('/api/status', {cache: 'no-store'});
        const data = await response.json();
        const people = data.people || [];
        const inCount = people.filter(p => p.status === 'in').length;
        document.querySelector('#summary').textContent = `401 在室 ${inCount}人 / 登録 ${people.length}人`;
        document.querySelector('#people').innerHTML = people.length
          ? people.map(p => `<section class="person ${p.status === 'out' ? 'out' : ''}">
              ${escapeHtml(names[p.name] || p.name)}
              <div class="status">${p.status === 'out' ? '外出 / 帰宅' : '在室'}</div>
            </section>`).join('')
          : '<p>現在、在室記録はありません。</p>';
        document.querySelector('#updated').textContent = `最終確認: ${new Date().toLocaleString('ja-JP')}`;
      } catch (error) {
        document.querySelector('#summary').textContent = 'サーバーから状態を取得できません';
      }
    }
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
""".replace("__NAME_MAP__", json.dumps(NAME_MAP, ensure_ascii=False))


def empty_state():
    return {"date": datetime.date.today().isoformat(), "people": []}


def load_state():
    today = datetime.date.today().isoformat()
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        state = empty_state()
    if state.get("date") != today:
        state = empty_state()
        save_state(state)
    return state


def save_state(state):
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_status(event, name):
    with STATE_LOCK:
        state = load_state()
        person = next((item for item in state["people"] if item["name"] == name), None)
        if event == "entry":
            if person:
                person["status"] = "in"
            else:
                state["people"].append({"name": name, "status": "in"})
        elif event == "exit" and person and person["status"] == "in":
            person["status"] = "out"
        save_state(state)
        return state


def blink_led(event):
    path = "/light4/blink" if event == "entry" else "/light5/blink"
    try:
        request = Request(
            LED_SERVER + path,
            headers={"ngrok-skip-browser-warning": "true"},
        )
        urlopen(request, timeout=3).close()
    except (URLError, TimeoutError, OSError):
        pass


class StatusHandler(BaseHTTPRequestHandler):
    def send_bytes(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_bytes(status, "application/json; charset=utf-8", body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.send_bytes(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif self.path == "/api/status":
            with STATE_LOCK:
                state = load_state()
            self.send_json(200, state)
        else:
            self.send_json(404, {"status": "not_found"})

    def do_POST(self):
        if self.path != "/api/events":
            self.send_json(404, {"status": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length))
            event = data.get("event")
            name = data.get("name")
            if event not in {"entry", "exit"} or not isinstance(name, str) or not name:
                raise ValueError("event must be entry/exit and name is required")
            state = update_status(event, name)
            threading.Thread(target=blink_led, args=(event,), daemon=True).start()
            self.send_json(200, {"status": "ok", "type": event, "data": state["people"]})
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(400, {"status": "error", "message": str(error)})

    def log_message(self, format, *args):
        print(f"[web] {self.address_string()} - {format % args}")


def start_camera_processes():
    env = os.environ.copy()
    env["STATUS_API_URL"] = f"http://127.0.0.1:{PORT}/api/events"
    return [
        subprocess.Popen([sys.executable, str(PROJECT_DIR / script)], cwd=PROJECT_DIR, env=env)
        for script in ("main_entry.py", "main_exit.py")
    ]


def stop_processes(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def main():
    server = ThreadingHTTPServer((HOST, PORT), StatusHandler)
    processes = []
    try:
        processes = start_camera_processes()
        print(f"Web page: http://127.0.0.1:{PORT}")
        print("Press Ctrl+C to stop the server and both camera processes.")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        server.server_close()
        stop_processes(processes)


if __name__ == "__main__":
    main()
