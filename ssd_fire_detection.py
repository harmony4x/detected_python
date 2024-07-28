import cv2
import tensorflow as tf

# Load SSD model
ssd_model = tf.saved_model.load('ssd_mobilenet_v2_fpnlite_320x320/saved_model')

def detect_fire_ssd(frame, model):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    # Process detections
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)
    scores = detections['detection_scores'][0].numpy()

    for i in range(len(scores)):
        if scores[i] > 0.5 and classes[i] == 1:  # Assuming class 1 is fire
            box = boxes[i] * [height, width, height, width]
            (startY, startX, endY, endX) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return frame

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    frame = detect_fire_ssd(frame, ssd_model)

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
