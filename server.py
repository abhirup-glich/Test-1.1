from flask import Flask, request, jsonify
from flask_cors import CORS
from logic import (
    connect_db,
    setup_db,
    process_web_image,
    mark_attendance,
    register_student_web
)

app = Flask(__name__)
CORS(app)

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/attendance", methods=["POST"])
def attendance():
    data = request.json
    emb = process_web_image(data.get("image"))

    if emb is None:
        return jsonify({"status": "no_face"})

    conn, cur = connect_db()
    result = mark_attendance(cur, emb)
    conn.close()

    return jsonify(result)

@app.route("/register", methods=["POST"])
def register():
    data = request.json

    conn, cur = connect_db()
    result = register_student_web(
        cur,
        data["roll"],
        data["name"],
        data["course"],
        data["images"]
    )
    conn.close()

    return jsonify(result)

if __name__ == "__main__":
    conn, cur = connect_db()
    setup_db(cur)
    conn.close()

    app.run(host="0.0.0.0", port=10000)
