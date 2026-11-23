from ultralytics import YOLO

# 모델 학습
def main() :
    model = YOLO("yolo26n.pt")  # 사전 학습된 모델을 불러오기
    results = model.train(data="datasets/ml-hw2-v2/data.yaml", epochs=100, imgsz=640)

# 환경이 윈도우라 가드 추가
if __name__ == "__main__":
    main()