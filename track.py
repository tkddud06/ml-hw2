from collections import defaultdict

import cv2
import numpy as np
import csv

from ultralytics import YOLO

def main():
    # YOLO26 모델을 불러오기
    model = YOLO("runs/detect/train/weights/best.pt")

    # 비디오 파일을 열기
    video_path = "track/20261122_200221_1.mp4"
    cap = cv2.VideoCapture(video_path)

    # 비디오 속성 받기
    fps = cap.get(cv2.CAP_PROP_FPS)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 비디오 저장
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("result.mp4", fourcc, fps, (vw, vh))

    # 트랙 히스토리를 저장
    track_history = defaultdict(list)

    # 추적선 csv로 저장
    with open("result.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "track_id", "class", "cx", "cy" ])

        frame_idx = 0

        # 비디오 프레임 순회
        while cap.isOpened():
            # 비디오에서 프레임 읽기
            success, frame = cap.read()

            if success:
                # 프레임에서 YOLO26 추적을 실행, 프레임 간 트랙을 유지
                result = model.track(frame, persist=True, conf=0.1, iou=0.5, tracker="botsort-custom.yaml")[0]

                # 박스와 트랙 ID, 클래스 정보 가져오기
                if result.boxes and result.boxes.is_track:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    classes = result.boxes.cls.int().cpu().tolist()

                    # 프레임에 결과를 시각화
                    frame = result.plot()

                    # 트랙을 그리기
                    for box, track_id, cls in zip(boxes, track_ids, classes):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y 중심점
                        if len(track) > 60:  # 60프레임 동안 트랙 60개를 유지
                            track.pop(0)

                        # 추적선을 그리고 csv로 별도로 저장
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 255), thickness=4)
                        writer.writerow([frame_idx, track_id, result.names[cls] , float(x), float(y)])

                # 주석이 추가된 프레임을 표시
                cv2.imshow("YOLO26 Tracking", frame)
                # 'q'를 누르면 루프를 종료
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                out.write(frame)
            else:
                # 비디오 끝에 도달하면 루프를 종료
                break
            frame_idx += 1

        # 비디오 캡처, 저장 객체를 해제하고 표시 창 닫기
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()