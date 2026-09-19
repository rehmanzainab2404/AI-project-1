import cv2
import numpy as np
import os
import json
import threading
import csv
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from insightface.app import FaceAnalysis
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ==========================================
# 1. AI ENGINE & STATE MEMORY
# ==========================================
print("[INFO] Initializing InsightFace (Buffalo_L) on CPU...")
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

# Global Memory State
known_embeddings = []
known_names = []
master_list = set()
session_detected = set()

camera_active = False
ai_lock = threading.Lock()

def train_model():
    """Reads dataset/ directory, extracts embeddings, trains in-memory."""
    global known_embeddings, known_names, master_list
    
    dataset_path = "dataset"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    new_embeddings = []
    new_names = []
    new_master_list = set()

    print("\n[INFO] Scanning dataset directory for training...")
    with ai_lock:
        for person_name in os.listdir(dataset_path):
            person_dir = os.path.join(dataset_path, person_name)
            
            if os.path.isdir(person_dir):
                new_master_list.add(person_name)
                
                for filename in os.listdir(person_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_dir, filename)
                        img = cv2.imread(img_path)
                        
                        if img is None: continue
                        
                        faces = app.get(img)
                        if len(faces) > 0:
                            new_embeddings.append(faces[0].embedding)
                            new_names.append(person_name)

        known_embeddings = new_embeddings
        known_names = new_names
        master_list = new_master_list
        print(f"[SUCCESS] Model trained! Knows {len(master_list)} people.\n")

def match_face(target_embedding, threshold=0.35):
    """Compares live face against memory."""
    if not known_embeddings:
        return "Unknown"
        
    sims = [np.dot(target_embedding, db_emb) / (np.linalg.norm(target_embedding) * np.linalg.norm(db_emb)) 
            for db_emb in known_embeddings]
            
    idx = np.argmax(sims)
    if sims[idx] > threshold:
        return known_names[idx]
    return "Unknown"

# ==========================================
# 2. AUTO-RETRAINING OBSERVER (WATCHDOG)
# ==========================================
train_timer = None

class DatasetWatcher(FileSystemEventHandler):
    def on_any_event(self, event):
        global train_timer
        if event.is_directory or event.src_path.endswith(('.png', '.jpg', '.jpeg')):
            if train_timer:
                train_timer.cancel()
            train_timer = threading.Timer(1.5, train_model)
            train_timer.start()

observer = Observer()
observer.schedule(DatasetWatcher(), path="dataset", recursive=True)
observer.start()

# ==========================================
# 3. PURE PYTHON HTTP & MJPEG SERVER
# ==========================================
class RequestHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        global camera_active, session_detected

        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('index.html', 'rb') as f:
                self.wfile.write(f.read())
                
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = {"detected": list(session_detected)}
            self.wfile.write(json.dumps(data).encode('utf-8'))
            
        elif self.path.startswith('/video_feed'):
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            camera_active = True
            session_detected.clear()
            cap = cv2.VideoCapture(0)
            
            # --- FPS HACK VARIABLES ---
            frame_count = 0
            frame_skip = 4  # Run AI every 4th frame (Adjust this if you want it faster/slower)
            last_known_faces = [] # Remembers the boxes for the skipped frames
            
            try:
                while camera_active:
                    success, frame = cap.read()
                    if not success: break
                    
                    frame_count += 1
                    
                    # Only run the heavy AI math every 4th frame
                    if frame_count % frame_skip == 0:
                        with ai_lock:
                            faces = app.get(frame)
                            last_known_faces = [] # clear old boxes
                            
                            for face in faces:
                                name = match_face(face.embedding)
                                if name != "Unknown":
                                    session_detected.add(name)
                                    
                                bbox = face.bbox.astype(int)
                                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                                # Save the box data
                                last_known_faces.append((bbox, name, color))

                    # Draw the boxes on EVERY frame so it looks smooth
                    for bbox, name, color in last_known_faces:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        cv2.putText(frame, name, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                pass 
            finally:
                cap.release()

    def do_POST(self):
        global camera_active, session_detected, master_list
        if self.path == '/stop':
            camera_active = False
            
            # 1. Calculate Logic
            detected_list = list(session_detected)
            absent_list = list(master_list - session_detected)
            
            # 2. GENERATE EXCEL (CSV) FILE
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"Attendance_{timestamp}.csv"
            
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Attendance Status", "Time Generated"])
                
                # Write Present people
                for person in detected_list:
                    writer.writerow([person, "Present", timestamp])
                    
                # Write Absent people
                for person in absent_list:
                    writer.writerow([person, "Absent", timestamp])
                    
            print(f"\n[EXCEL GENERATED] Saved attendance to {filename}")

            # 3. Send final summary to UI
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            summary = {"detected": detected_list, "absent": absent_list}
            self.wfile.write(json.dumps(summary).encode('utf-8'))

# ==========================================
# 4. INITIALIZATION
# ==========================================
if __name__ == "__main__":
    train_model() 
    
    server_address = ('', 8080)
    httpd = ThreadingHTTPServer(server_address, RequestHandler)
    
    print("\n[READY] Server running on http://localhost:8080")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        observer.stop()
        observer.join()
        httpd.server_close() 
