import re
import cv2
import numpy as np
import pytesseract
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, render_template, request, send_file
import os

app = Flask(__name__)

@app.route('/text_file')
def get_text_file():
    return send_file('./car_plate_data.txt', as_attachment=True)

@app.route('/')
def index():
    folder_path = "Plate-Number-Classification/CroppedPlat"
    files = os.listdir(folder_path)
    return render_template('index.html', files=files)

@app.route('/button_click', methods=['POST'])
def handler():
    video_file = request.files['video_file']
    video_file_path = 'uploaded_video.mp4'  # Save the video file temporarily
    video_file.save(video_file_path)
    Predict(video_file_path)
    return "Video processing started."

def Predict(path, gaussian_kernel_size=(3, 3), median_kernel_size=5, contrast_stretch_percentiles=(1, 99), thresholding_method=cv2.THRESH_BINARY + cv2.THRESH_OTSU):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    model = YOLO('PlateNumberModel/best.pt')

    cap = cv2.VideoCapture(path)

    with open("PlateNumberModel/coco1.txt", "r") as my_file:
        class_list = my_file.read().split("\n")

    area = [(27, 150), (16, 300), (1015, 300), (992, 150)]

    count = 0
    processed_numbers = set()

    output_folder = "Plate-Number-Classification/CroppedPlat"
    preprocessed_folder = "Plate-Number-Classification/Preprocessed"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(preprocessed_folder, exist_ok=True)

    image_counter = 1
    count_file_path = "Plate-Number-Classification/count.txt"
    if os.path.exists(count_file_path):
        with open(count_file_path, "r") as count_file:
            image_counter = int(count_file.read().strip())

    has_written = False

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
                
                # Convert to grayscale
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # cv2.imwrite(f"{preprocessed_folder}/step1_gray_{image_counter}.png", gray)

                # gray = cv2.bilateralFilter(gray, 10, 20, 20)
                # cv2.imwrite(f"{preprocessed_folder}/step2_gray_{image_counter}.png", gray)

                # Apply Gaussian Blur
                # gray = cv2.GaussianBlur(gray, gaussian_kernel_size, 0)
                # cv2.imwrite(f"{preprocessed_folder}/step2_gaussian_{image_counter}.png", gray)
                
                # Apply Median Blur
                # gray = cv2.medianBlur(gray, median_kernel_size)
                # cv2.imwrite(f"{preprocessed_folder}/step3_median_{image_counter}.png", gray)
                
                # Apply contrast stretching first
                # p1, p99 = np.percentile(gray, contrast_stretch_percentiles)
                # gray = np.clip((gray - p1) * (255.0 / (p99 - p1)), 0, 255).astype(np.uint8)
                # cv2.imwrite(f"{preprocessed_folder}/step4_contrast_{image_counter}.png", gray)
                
                # Apply thresholding before histogram equalization
                _, thresholded = cv2.threshold(gray, 0, 255, thresholding_method)
                cv2.imwrite(f"{preprocessed_folder}/step5_threshold_{image_counter}.png", thresholded)
                
                # Apply histogram equalization after thresholding
                # equalized = cv2.equalizeHist(gray)
                # cv2.imwrite(f"{preprocessed_folder}/step6_histogram_{image_counter}.png", equalized)
                
                # Extract text using Tesseract OCR
                text = pytesseract.image_to_string(gray).strip()
                
                # Post-process the extracted text
                text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
                
                if text and text[0].islower():
                    continue
                
                print(text)
                if text not in processed_numbers and not has_written:
                    processed_numbers.add(text) 
                    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open("Plate-Number-Classification/car_plate_data.txt", "a") as file:
                        file.write(f"{text}\t{current_datetime}\n")
                    
                    filename = f"{output_folder}/Pict{image_counter}.png"
                    cv2.imwrite(filename, crop)
                    image_counter += 1
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imshow('crop', crop)
                    
                    has_written = True

        cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
        cv2.imshow("RGB", frame)
        cv2.waitKey(1)

    with open(count_file_path, "w") as count_file:
        count_file.write(str(image_counter))

    cap.release()    
    cv2.destroyAllWindows()
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
