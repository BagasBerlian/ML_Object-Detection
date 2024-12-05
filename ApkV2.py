# pylint: disable=no-member
# pylint: disable=no-name-in-module
import cv2
import numpy as np
import threading
import queue
import tkinter as tk
from tkinter import messagebox
import mysql.connector
from datetime import datetime
import os

# Path konfigurasi YOLO
CONFIG_FILE = "ML_ObjectDetection/yolov4-p5.cfg"
WEIGHTS_FILE = "ML_ObjectDetection/yolov4-p5.weights"
NAMES_FILE = "ML_ObjectDetection/coco.names"
NEW_OBJECT_DIR = "ML_ObjectDetection/New_Object"

# Membaca kelas dari COCO dataset
with open(NAMES_FILE, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Inisialisasi YOLO
net = cv2.dnn.readNet(WEIGHTS_FILE, CONFIG_FILE)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Konfigurasi kamera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 896)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 896)

# Queue untuk komunikasi antar thread
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

# Membuat direktori new_object jika belum ada
os.makedirs(NEW_OBJECT_DIR, exist_ok=True)


def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indices


def detection_thread():
    # Thread untuk menjalankan deteksi objek secara paralel.
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = detect_objects(frame)
            if not result_queue.full():
                result_queue.put(results)


def save_to_database(classname, filename):
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="object_database"
        )
        cursor = connection.cursor()
        query = "INSERT INTO objects (classname, filename, created_at) VALUES (%s, %s, %s)"
        datetime_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(query, (classname, filename, datetime_now))
        connection.commit()
        cursor.close()
        connection.close()
        messagebox.showinfo("Success", f"Data '{classname}' berhasil disimpan ke database!")
    except mysql.connector.Error as err:
        messagebox.showerror("Error", f"Database error: {err}")


def save_image_to_folder(frame, classname):
    filename = f"{classname}.jpg"
    img_path = os.path.join(NEW_OBJECT_DIR, filename)
    cv2.imwrite(img_path, frame)
    print(f"Gambar '{filename}' disimpan di {NEW_OBJECT_DIR}")
    return filename


def show_input_frame(frame):
    def on_submit():
        classname = name_entry.get().strip()
        if classname:
            filename = save_image_to_folder(frame, classname)
            save_to_database(classname, filename)
            input_window.destroy()
        else:
            messagebox.showwarning("Input Error", "Nama objek tidak boleh kosong.")

    # Membuat frame GUI
    input_window = tk.Tk()
    input_window.title("Masukkan Nama Objek")
    input_window.geometry("300x150")

    tk.Label(input_window, text="Nama Objek Baru:").pack(pady=10)
    name_entry = tk.Entry(input_window, width=30)
    name_entry.pack(pady=5)

    tk.Button(input_window, text="Submit", command=on_submit).pack(pady=10)

    input_window.mainloop()


def main():
    threading.Thread(target=detection_thread, daemon=True).start()

    frame_count = 0
    skip_frames = 1
    last_results = None
    confidence_threshold = 0.80

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % skip_frames == 0 and frame_queue.empty():
            frame_queue.put(frame.copy())

        if not result_queue.empty():
            last_results = result_queue.get()

        if last_results:
            boxes, confidences, class_ids, indices = last_results
            low_confidence_detected = False

            for i in indices:
                i = i[0] if isinstance(i, (list, np.ndarray)) else i
                if confidences[i] >= confidence_threshold:
                    x, y, w, h = boxes[i]
                    label = f"{classes[class_ids[i]]}: {int(confidences[i] * 100)}%"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 50, 255), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 50, 255), 2)
                else:
                    low_confidence_detected = True

            if low_confidence_detected:
                cv2.putText(frame, "Objek baru terdeteksi...(Tekan 'S')", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Object Detection with YOLO", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):  # Tekan 'Q' untuk keluar
            break
        elif key & 0xFF == ord('s') and last_results:
            print("Key 'S' pressed: Input object data...")
            show_input_frame(frame)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
