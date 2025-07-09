import argparse
import cv2
import numpy as np
import time
from ultralytics import YOLO
from motrackers import CentroidTracker

def distance_estimation(input_path, output_path, model_path):
    model = YOLO(model_path) #Load Model

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f'Error opening video file : {input_path}')
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    #Tracking
    tracker = CentroidTracker(max_lost=5, tracker_output_format='mot_challenge')
    track_data = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        results = model(frame)

        detections = []
        confidences = []
        class_ids = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                w, h = x2 - x1, y2 - y1
                detections.append([x1, y1, w, h])
                confidences.append(conf[i])
                class_ids.append(cls[i])

        tracks = tracker.update(np.array(detections), np.array(confidences), np.array(class_ids))

        # Distance Estimation Logic
        for trk in tracks:
            track_id = int(trk[1])
            xmin, ymin, w, h = map(int, trk[2:6])
            cx = xmin + w // 2
            cy = ymin + h // 2

            if track_id not in track_data:
                track_data[track_id] = {'centroid': (cx, cy), 'total_distance': 0.0}

            prev_cx, prev_cy = track_data[track_id]['centroid']
            dx, dy = cx - prev_cx, cy - prev_cy
            dist = np.sqrt(dx**2 + dy**2)

            if dist > 30:
                dist = 0

            track_data[track_id]['total_distance'] += dist
            track_data[track_id]['centroid'] = (cx, cy)

            tdp = round(track_data[track_id]['total_distance'], 1)
            label = f"ID {track_id} | TDP: {tdp}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (0, 255, 0), 1)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        out.write(frame)
        frame_count += 1
        print(f"Processed frame {frame_count} | Time: {time.time() - start_time:.3f}s")

    cap.release()
    out.release()
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Distance Estimation using YOLOv11s")
    parser.add_argument('--video_input', type=str, required=True, help='Path to input video')
    parser.add_argument('--video_output', type=str, required=True, help='Path to save output video')
    parser.add_argument('--model_path', type=str, required=True, help='Path to YOLOv11s .pt model file')
    args = parser.parse_args()

    distance_estimation(args.video_input, args.video_output, args.model_path)