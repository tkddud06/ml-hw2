from ultralytics import YOLO

def main():

    # 모델 로드
    model = YOLO("runs/detect/train/weights/best.pt")  # 사용자 지정 모델 로드

    # 모델을 검증
    metrics = model.val(data="datasets/ml-hw2-v2/data.yaml", split="test")

# 환경이 윈도우라 가드 추가
if __name__ == "__main__":
    main()