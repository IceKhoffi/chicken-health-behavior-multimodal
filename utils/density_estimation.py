import argparse
import cv2
import numpy as np
import time
import os
import scipy.spatial as spatial
from ultralytics import YOLO
from motrackers import CentroidTracker

def density_estimation(input_path, output_path, model_path):
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
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                detections.append([x1, y1, w, h])
                confidences.append(scores[i])
                class_ids.append(classes[i])
        
        # Update tracker
        tracks = tracker.update(np.array(detections), np.array(confidences), np.array(class_ids))
        all_centroids = []

        for trk in tracks:
            track_id = int(trk[1])
            xmin, ymin, w, h = map(int, trk[2:6])
            cx, cy = xmin + w // 2, ymin + h // 2
            all_centroids.append((cx, cy))

            if track_id not in track_data:
                track_data[track_id] = {'centroid': (cx, cy), 'total_distance': 0.0}

            prev_cx, prev_cy = track_data[track_id]['centroid']
            dx, dy = cx - prev_cx, cy - prev_cy
            dist = np.hypot(dx, dy)

            if dist > 30:
                dist = 0

            track_data[track_id]['total_distance'] += dist
            track_data[track_id]['centroid'] = (cx, cy)
        
        # Density Logic
        if len(all_centroids) > 1:
            points = np.array(all_centroids)
            tree = spatial.KDTree(points)
            radius = 100  # in pixels
            neighbors = tree.query_ball_tree(tree, radius)
            density = np.array([len(n) for n in neighbors])
            dense_indices = np.where(density >= 6)[0]

            for i in dense_indices:
                x, y = points[i]
                cv2.rectangle(frame, (x - 30, y - 30), (x + 30, y + 30), (0, 0, 255), 1)
                cv2.putText(frame, 'High Dense', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # Write frame
        out.write(frame)
        print(f"Processed frame | Time: {time.time() - start_time:.3f}s")

    cap.release()
    out.release()
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Density Estimation using YOLOv11s")
    parser.add_argument('--video_input', type=str, required=True, help='Path to input video')
    parser.add_argument('--video_output', type=str, required=True, help='Path to save output video')
    parser.add_argument('--model_path', type=str, required=True, help='Path to YOLOv11s .pt model file')
    args = parser.parse_args()

    density_estimation(args.video_input, args.video_output, args.model_path)