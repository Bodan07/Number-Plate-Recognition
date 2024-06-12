import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import pytesseract
from datetime import datetime
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO('PlateNumberModel/best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('PlateNumberModel/nuik.mp4')

my_file = open("PlateNumberModel/coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

area = [(27, 150), (16, 200), (1015, 200), (992, 150)]

count = 0
list1 = []
processed_numbers = set()

# Create a folder to save cropped frames if it doesn't exist
output_folder = "Plate-Number-Classification/cropped_frames"
os.makedirs(output_folder, exist_ok=True)

# Counter for naming the images
image_counter = 1

# Load the current count from a file if it exists
count_file_path = "Plate-Number-Classification/count.txt"
if os.path.exists(count_file_path):
    with open(count_file_path, "r") as count_file:
        image_counter = int(count_file.read().strip())

while True:    
    ret, frame = cap.read()
    if not ret:
        break
   
    count += 1
    if count % 3 != 0:
        continue
   
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
   
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
        if result >= 0:
            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 10, 20, 20)
            text = pytesseract.image_to_string(gray).strip()
            text = text.replace('(', '').replace(')', '').replace(',', '').replace(']','')
            print(text)
            if text not in processed_numbers:
                processed_numbers.add(text) 
                list1.append(text)
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("Plate-Number-Classification/car_plate_data.txt", "a") as file:
                    file.write(f"{text}\t{current_datetime}\n")
                
                # Save the cropped frame with a numbered filename
                filename = f"{output_folder}/Pict{image_counter}.png"
                cv2.imwrite(filename, crop)
                image_counter += 1
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('crop', crop)

    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    cv2.imshow("RGB", frame)
    cv2.waitKey(1)

# Save the current count to the file
with open(count_file_path, "w") as count_file:
    count_file.write(str(image_counter))

cap.release()    
cv2.destroyAllWindows()
