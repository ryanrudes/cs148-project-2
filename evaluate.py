import numpy as np
import torch
import cv2

from resnext import ResNeXt

device = torch.device("mps")

model = ResNeXt(
    #layers=[3, 4, 23, 3],
    layers=[3, 4, 23, 3],
    num_classes=10,
    groups=64,
    width_per_group=4,
)

model = model.to(device)

# Load checkpoint
checkpoint = torch.load("best (1).pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Stream video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize, convert to tensor, normalize)
    img = cv2.resize(frame, (384, 384))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = np.copy(img)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
    img_tensor = img_tensor.to(device)

    # Predict the digit
    with torch.no_grad():
        logits = model(img_tensor)
        pred = torch.argmax(logits, dim=1).item()

    # Display the prediction on the frame
    #cv2.putText(frame, f"Predicted: {pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Digit Recognition", frame)
    print(pred)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
