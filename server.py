# server.py
from flask import Flask, request, jsonify, send_from_directory, Blueprint
from flask_cors import CORS
import os
from datetime import datetime, timedelta
import subprocess
import sys
import json
from Logic import logic  # Import your logic module
from werkzeug.utils import secure_filename
from Logic.logic import connect_db, setup_db, mark_attendance
from Logic import logic as logic_mod
import logging
 
app = Flask(__name__, static_folder='WebPage', static_url_path='/WebPage')
api = Blueprint('api', __name__, url_prefix='/api')
cors = CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5000", "http://127.0.0.1:5000"]}}, supports_credentials=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

@app.after_request
def add_security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-XSS-Protection"] = "1; mode=block"
    path = request.path.lower()
    if path.endswith('.js') or path.endswith('.css'):
        resp.headers["Cache-Control"] = "public, max-age=604800"
    elif path.endswith('.html'):
        resp.headers["Cache-Control"] = "no-cache"
    return resp

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'webm', 'mp4', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/set_cookie", methods=["POST", "OPTIONS"])
def set_cookie():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(silent=True) or {}
    name = data.get("name", "fast_auth")
    value = data.get("value", "token")
    max_age = data.get("max_age", 3600)
    secure = bool(data.get("secure", False))
    resp = jsonify({"ok": True})
    resp.set_cookie(
        key=name,
        value=value,
        max_age=max_age,
        expires=timedelta(seconds=max_age),
        httponly=True,
        samesite="None",
        secure=secure,
        domain="localhost",
        path="/",
    )
    return resp

@app.route("/echo_cookie", methods=["GET", "OPTIONS"])
def echo_cookie():
    if request.method == "OPTIONS":
        return ("", 204)
    name = request.args.get("name", "fast_auth")
    val = request.cookies.get(name)
    return jsonify({"cookie": {name: val}}), 200

def upload_video_impl():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save the file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"attendance_{timestamp}.webm"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Call the face recognition logic
            conn, cur = connect_db()
            result = mark_attendance(cur, video_path=filepath)
            cur.close()
            conn.close()
            
            if result.get('status') == 'success':
                return jsonify({
                    'message': 'Attendance marked successfully',
                    'data': {
                        'name': result['name'],
                        'roll': result['roll'],
                        'time': result['time'],
                        'confidence': result['confidence']
                    }
                })
            else:
                return jsonify({
                    'status': result.get('status', 'error'),
                    'message': 'No students found' if result.get('status') == 'no_student' else result.get('error', 'Attendance not marked'),
                    'time': result.get('time')
                }), 200 if result.get('status') in ['no_student', 'no_face'] else 400
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
 
@api.route('/upload', methods=['POST'])
def api_upload_video():
    return upload_video_impl()
 
@app.route('/upload', methods=['POST'])
def upload_video():
    return upload_video_impl()

@app.route('/api/upload', methods=['POST'])
def api_upload_video_direct():
    return upload_video_impl()

def check_attendance_impl():
    try:
        conn, cur = logic.connect_db()
        cur.execute("""
            SELECT s.roll, s.name, s.course, a.time, a.confidence 
            FROM attendance a
            JOIN students s ON a.roll = s.roll
            ORDER BY a.time DESC
        """)
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert datetime to string for JSON serialization
        attendance_list = []
        for row in results:
            row_dict = dict(row)
            row_dict['time'] = row_dict['time'].isoformat() if row_dict['time'] else None
            attendance_list.append(row_dict)
            
        return jsonify({
            'status': 'success',
            'attendance': attendance_list
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
 
@api.route('/check_attendance', methods=['GET'])
def api_check_attendance():
    return check_attendance_impl()
 
@app.route('/check_attendance', methods=['GET'])
def check_attendance():
    return check_attendance_impl()

@app.route('/api/check_attendance', methods=['GET'])
def api_check_attendance_direct():
    return check_attendance_impl()

def register_student_impl():
    try:
        data = request.json
        roll = data.get('roll')
        name = data.get('name')
        course = data.get('course')
        images = data.get('images') # Expects {center: '...', left: '...', right: '...'}
        
        if not all([roll, name, course, images]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        conn, cur = logic.connect_db()
        result = logic.register_student_web(cur, roll, name, course, images)
        conn.close()
        
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
@api.route('/register_student', methods=['POST'])
def api_register_student():
    return register_student_impl()
 
@app.route('/register_student', methods=['POST'])
def register_student():
    return register_student_impl()

@app.route('/api/register_student', methods=['POST'])
def api_register_student_direct():
    return register_student_impl()

def delete_last_attendance_impl():
    try:
        conn, cur = logic.connect_db()
        deleted = logic.delete_last_attendance(cur)
        conn.close()
        
        if deleted:
            return jsonify({
                'status': 'success',
                'message': f"Deleted attendance for {deleted['name']} at {deleted['time']}"
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'No attendance record found to delete'
            }), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
@api.route('/delete_last_attendance', methods=['POST'])
def api_delete_last_attendance_route():
    return delete_last_attendance_impl()
 
@app.route('/delete_last_attendance', methods=['POST'])
def delete_last_attendance_route():
    return delete_last_attendance_impl()

@app.route('/api/delete_last_attendance', methods=['POST'])
def api_delete_last_attendance_direct():
    return delete_last_attendance_impl()

def clear_all_attendance_impl():
    try:
        conn, cur = logic.connect_db()
        success = logic.clear_all_attendance(cur)
        conn.close()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'All attendance records cleared'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to clear attendance'
            }), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
@api.route('/clear_all_attendance', methods=['POST'])
def api_clear_all_attendance_route():
    return clear_all_attendance_impl()
 
@app.route('/clear_all_attendance', methods=['POST'])
def clear_all_attendance_route():
    return clear_all_attendance_impl()

@app.route('/api/clear_all_attendance', methods=['POST'])
def api_clear_all_attendance_direct():
    return clear_all_attendance_impl()

def get_students_impl():
    try:
        conn, cur = logic.connect_db()
        students = logic.fetch_all_students(cur)
        conn.close()
        return jsonify({'students': students})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
@api.route('/students', methods=['GET'])
def api_get_students():
    return get_students_impl()
 
@app.route('/students', methods=['GET'])
def get_students():
    return get_students_impl()

# Explicit /api routes without blueprint (compat)
@app.route('/api/students', methods=['GET'])
def api_students_direct():
    return get_students_impl()

def class_start_impl():
    try:
        data = request.json if request.is_json else {}
        course = data.get('course')
        notes = data.get('notes')
        conn, cur = logic.connect_db()
        result = logic.record_class_start(cur, course, notes)
        conn.close()
        if result.get('status') == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500
 
@api.route('/class_start', methods=['POST'])
def api_class_start():
    return class_start_impl()
 
@app.route('/class_start', methods=['POST'])
def class_start():
    return class_start_impl()

@app.route('/api/class_start', methods=['POST'])
def api_class_start_direct():
    return class_start_impl()

def update_student_impl(roll):
    try:
        data = request.json
        conn, cur = logic.connect_db()
        success = logic.update_student_record(cur, roll, data.get('name'), data.get('course'))
        conn.close()
        
        if success:
            return jsonify({'status': 'success', 'message': 'Student updated'})
        else:
            return jsonify({'status': 'error', 'message': 'Update failed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
@api.route('/students/<roll>', methods=['PUT'])
def api_update_student(roll):
    return update_student_impl(roll)
 
@app.route('/students/<roll>', methods=['PUT'])
def update_student(roll):
    return update_student_impl(roll)

@app.route('/api/students/<roll>', methods=['PUT'])
def api_update_student_direct(roll):
    return update_student_impl(roll)

def delete_student_impl(roll):
    try:
        conn, cur = logic.connect_db()
        success = logic.delete_student_record(cur, roll)
        conn.close()
        
        if success:
            return jsonify({'status': 'success', 'message': 'Student deleted'})
        else:
            return jsonify({'status': 'error', 'message': 'Delete failed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
@api.route('/students/<roll>', methods=['DELETE'])
def api_delete_student(roll):
    return delete_student_impl(roll)
 
@app.route('/students/<roll>', methods=['DELETE'])
def delete_student(roll):
    return delete_student_impl(roll)

@app.route('/api/students/<roll>', methods=['DELETE'])
def api_delete_student_direct(roll):
    return delete_student_impl(roll)

@app.route('/')
def root_index():
    return send_from_directory(os.path.join(app.root_path, 'WebPage'), 'FAST V2.html')

@app.route('/admin')
def admin_page():
    return send_from_directory(os.path.join(app.root_path, 'WebPage', 'Admin Page'), 'admin.html')

@app.route('/Admin')
def admin_page_caps():
    return send_from_directory(os.path.join(app.root_path, 'WebPage', 'Admin Page'), 'admin.html')

@app.route('/admin.html')
def admin_html_route():
    return send_from_directory(os.path.join(app.root_path, 'WebPage', 'Admin Page'), 'admin.html')

@app.route('/serveradmin')
def server_admin_page():
    return send_from_directory(os.path.join(app.root_path, 'WebPage', 'Admin Page'), 'serveradmin.html')

@app.route('/serveradmin.html')
def server_admin_html_route():
    return send_from_directory(os.path.join(app.root_path, 'WebPage', 'Admin Page'), 'serveradmin.html')

@app.route('/assets/<path:path>')
def assets(path):
    return send_from_directory(os.path.join(app.root_path, 'WebPage', 'assets'), path)

@app.route('/WebPage/<path:path>')
def serve_webpage_path(path):
    return send_from_directory(os.path.join(app.root_path, 'WebPage'), path)

@app.route('/Loginv2/<path:path>')
def login_assets(path):
    return send_from_directory(os.path.join(app.root_path, 'WebPage', 'Loginv2'), path)

@app.route('/Admin Page/<path:path>')
def admin_assets(path):
    return send_from_directory(os.path.join(app.root_path, 'WebPage', 'Admin Page'), path)

@app.route('/studentdashboard')
def student_dashboard():
    return send_from_directory(os.path.join(app.root_path, 'WebPage', 'Student page'), 'studentdashboard.html')

@app.route('/PrivacyPolicy.html')
def privacy_policy():
    return send_from_directory(os.path.join(app.root_path, 'WebPage'), 'PrivacyPolicy.html')

@app.route('/TermsOfService.html')
def terms_of_service():
    return send_from_directory(os.path.join(app.root_path, 'WebPage'), 'TermsOfService.html')

@app.route('/GetStarted.html')
def get_started():
    return send_from_directory(os.path.join(app.root_path, 'WebPage'), 'GetStarted.html')

@app.route('/LoginAs.html')
def login_as():
    return send_from_directory(os.path.join(app.root_path, 'WebPage'), 'LoginAs.html')

@app.route('/LearnMore.html')
def learn_more():
    return send_from_directory(os.path.join(app.root_path, 'WebPage'), 'LearnMore.html')


@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not Found'}), 404
    return send_from_directory(os.path.join(app.root_path, 'WebPage'), 'FAST V2.html')

@app.errorhandler(500)
def handle_500(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Server Error'}), 500
    return send_from_directory(os.path.join(app.root_path, 'WebPage'), 'FAST V2.html')

@app.before_request
def log_request():
    logging.info("%s %s", request.method, request.path)

app.register_blueprint(api)
for rule in app.url_map.iter_rules():
    print(str(rule))

@app.route('/api/<path:subpath>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def api_fallback(subpath):
    try:
        if subpath == 'upload' and request.method == 'POST':
            return upload_video_impl()
        if subpath == 'check_attendance' and request.method == 'GET':
            return check_attendance_impl()
        if subpath == 'register_student' and request.method == 'POST':
            return register_student_impl()
        if subpath == 'delete_last_attendance' and request.method == 'POST':
            return delete_last_attendance_impl()
        if subpath == 'clear_all_attendance' and request.method == 'POST':
            return clear_all_attendance_impl()
        if subpath == 'students' and request.method == 'GET':
            return get_students_impl()
        if subpath.startswith('students/') and request.method in ['PUT', 'DELETE']:
            roll = subpath.split('/', 1)[1]
            if request.method == 'PUT':
                return update_student_impl(roll)
            if request.method == 'DELETE':
                return delete_student_impl(roll)
        if subpath == 'class_start' and request.method == 'POST':
            return class_start_impl()
        return jsonify({'error': 'Not Found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug/routes', methods=['GET'])
def debug_routes():
    return jsonify({'routes': [str(rule) for rule in app.url_map.iter_rules()]})
if __name__ == '__main__':
    print("Starting server setup...")
    try:
        # Initialize database tables if they don't exist
        print("Connecting to database...")
        conn, cur = logic.connect_db()
        print("Setting up database...")
        logic.setup_db(cur)
        cur.close()
        conn.close()
        
# server.py (Indentation Correction)
# ...
        
        # Run the Flask app
        print("Starting Flask app...")
        if __name__ == "__main__": 
            app.run()  # <-- Must be indented!
    except Exception as e:
        print(f"Error starting server: {e}")
