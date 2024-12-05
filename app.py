# pylint: disable=no-member
# pylint: disable=no-name-in-module
import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("ML_ObjectDetection\\yolov4.weights", "ML_ObjectDetection\\yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# Load COCO class labels
with open("ML_ObjectDetection\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize camera
camera = cv2.VideoCapture(0)

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
            if confidence > 0.5:  # confidence threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:  # Memastikan ada deteksi
        for i in indices.flatten():  # Gunakan flatten() jika indices adalah array 2D
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {int(confidences[i] * 100)}%"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
frame_count = 0
def main():
    global frame_count
    while True:
        _, frame = camera.read()
        detect_objects(frame)
        cv2.imshow("Object Detection with YOLO", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera.release()
            cv2.destroyAllWindows()
            break

    

if __name__ == "__main__":
    main()