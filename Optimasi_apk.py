# pylint: disable=no-member
# pylint: disable=no-name-in-module
import cv2
import numpy as np
import threading
import queue


net = cv2.dnn.readNet("ML_ObjectDetection\\yolov4.weights", "ML_ObjectDetection\\yolov4.cfg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

with open("ML_ObjectDetection\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 608)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 608)

# Queues for thread communication
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indices

def detection_thread():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = detect_objects(frame)
            if not result_queue.full():
                result_queue.put(results)

# Start detection thread
threading.Thread(target=detection_thread, daemon=True).start()

frame_count = 0
skip_frames = 1 
last_results = None

def main():
    global frame_count, last_results
    while True:
        ret, frame = camera.read()
        camera_mirror = cv2.flip(frame,1)
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames == 0:
            if frame_queue.empty():
                frame_queue.put(camera_mirror.copy())

        if not result_queue.empty():
            last_results = result_queue.get()

        if last_results:
            boxes, confidences, class_ids, indices = last_results
            for i in indices:
                i = i[0] if isinstance(i, (list, np.ndarray)) else i
                x, y, w, h = boxes[i]
                print(f"Box {i}: x={x}, y={y}, w={w}, h={h}")
                label = f"{classes[class_ids[i]]}: {int(confidences[i] * 100)}%"
                cv2.rectangle(camera_mirror, (x, y), (x + w, y + h), (0, 50, 255), 2)
                cv2.putText(camera_mirror, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 50, 255), 2)
                
        cv2.imshow("Object Detection with YOLO", camera_mirror)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()