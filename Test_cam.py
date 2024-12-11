import cv2 as cv
import torch
from torchvision import transforms
from ultralytics import YOLO
from Model_RN_Att import ResNet18WithAttention  # Import model mới
from PIL import Image

# 1. Tải mô hình ResNet18WithAttention
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18WithAttention(num_classes=1)
model.load_state_dict(torch.load('./weight/as_model_0.880.pt', map_location=device))
model.to(device)
model.eval()

# 2. Biến đổi
tfms = transforms.Compose([
    # transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Tải mô hình YOLOv9
yolo_model = YOLO("./weight/50epoch.pt")  # Thay bằng đường dẫn mô hình YOLOv9

# 4. Mở camera
camera = cv.VideoCapture(0)

while cv.waitKey(1) & 0xFF != ord('q'):
    ret, img = camera.read()
    if not ret:
        break

    # 5. Dự đoán khuôn mặt bằng YOLOv9
    results = yolo_model(img)
    detections = results[0].boxes  # Lấy các bounding box

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Toạ độ bounding box
        confidence = box.conf[0]  # Độ chính xác của detection

        if confidence > 0.5:  # Ngưỡng confidence
            # 6. Cắt khuôn mặt từ ảnh
            # faceRegion = img[y1:y2, x1:x2]
            # faceRegion = cv.cvtColor(faceRegion, cv.COLOR_BGR2RGB)
            # faceRegion = tfms(faceRegion)
            # faceRegion = faceRegion.unsqueeze(0).to(device)
            faceRegion = img[y1:y2, x1:x2]  # Trích xuất khu vực khuôn mặt
            faceRegion = cv.cvtColor(faceRegion, cv.COLOR_BGR2RGB)  # Chuyển sang RGB
            faceRegion = Image.fromarray(faceRegion)  # Chuyển từ numpy.ndarray sang PIL.Image
            faceRegion = tfms(faceRegion)  # Áp dụng các phép biến đổi
            faceRegion = faceRegion.unsqueeze(0).to(device)  # Chuyển sang batch và đưa vào thiết bị

            # 7. Phát hiện spoofing với ResNet18WithAttention
            with torch.no_grad():
                prediction = model(faceRegion).sigmoid().item()  # Giá trị xác suất (0 - 1)
            p = 0.5 # Tự chọn
            label = 'Fake' if prediction < p else 'Real'
            color = (0, 0, 255) if prediction < 0.5 else (0, 255, 0)
            cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv.putText(img, f"{label}: {prediction:.2f}", (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv.imshow("YOLOv9 + ResNet18WithAttention Anti-Spoofing", img)
