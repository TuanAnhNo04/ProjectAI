from difflib import SequenceMatcher
import os
import csv
import sqlite3
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
import pandas as pd
from models import init_db, register_user, get_user
from nets import nn
from utils import util
from warnings import filterwarnings

filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = 'your_secret_key'
bcrypt = Bcrypt(app)

# Load the ONNX models for detection, recognition, and classification
try:
    detection = nn.Detection('./weights/detection.onnx')
    recognition = nn.Recognition('./weights/recognition.onnx')
    classification = nn.Classification('./weights/classification.onnx')
except Exception as e:
    raise RuntimeError("Error loading model weights") from e

# Initialize the database
init_db()

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Thiếu tên đăng nhập hoặc mật khẩu!"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    success = register_user(username, hashed_password)

    if success:
        return jsonify({"message": "Đăng ký thành công!"}), 201
    else:
        return jsonify({"error": "Tên đăng nhập đã tồn tại!"}), 409

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = get_user(username)
    if user and bcrypt.check_password_hash(user[2], password):
        return jsonify({"message": "Đăng nhập thành công!"}), 200
    else:
        return jsonify({"error": "Tên đăng nhập hoặc mật khẩu không đúng!"}), 401

@app.route('/logout', methods=['POST'])
def logout():
    return jsonify({"message": "Đăng xuất thành công!"}), 200

def save_results_to_csv(filepath, results, confidences, search_results):
    try:
        # Read the existing content to determine the next ID
        current_id = 0
        if os.path.exists('model_results.csv'):
            with open('model_results.csv', mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                current_id = sum(1 for row in reader)  # Count the number of existing rows for the next ID

        # Write to CSV file
        with open('model_results.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([current_id, filepath, len(results)] + results + confidences + search_results)  # Add search results
    except Exception as e:
        app.logger.error("Error saving to CSV: %s", e)
import csv
from difflib import SequenceMatcher

def search_medicine_in_csv(medicine_name):
    """Searches for a medicine by name in the CSV file and returns matching entries."""
    # Giả sử bạn đã đọc file CSV và lưu vào DataFrame
    df = pd.read_csv('project.csv')

    # Tìm kiếm tên thuốc chứa chuỗi medicine_name (không phân biệt chữ hoa chữ thường)
    search_results = df[df['Composition'].str.contains(medicine_name, case=False, na=False)]
    # Chuyển đổi DataFrame thành danh sách các từ điển
    return search_results.to_dict(orient='records')


def search_medicine_by_name(medicine_name):
    """Searches for a medicine by name in the CSV file and returns matching entries."""
    # Giả sử bạn đã đọc file CSV và lưu vào DataFrame
    df = pd.read_csv('project.csv')

    # Tìm kiếm tên thuốc chứa chuỗi medicine_name (không phân biệt chữ hoa chữ thường)
    search_results = df[df['Medicine_Name'].str.contains(medicine_name, case=False, na=False)]
    # Chuyển đổi DataFrame thành danh sách các từ điển
    return search_results.to_dict(orient='records')

@app.route('/search-medicine', methods=['GET'])
def search_medicine():
    medicine_name = request.args.get('name')
    if not medicine_name:
        return jsonify({"error": "Thiếu tên thuốc trong yêu cầu!"}), 400

    # Call the renamed function to search for the medicine
    search_results = search_medicine_by_name(medicine_name)
    
    if search_results:
        return jsonify({"results": search_results}), 200
    else:
        return jsonify({"message": "Không tìm thấy thông tin thuốc!"}), 404

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided!"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Invalid file name!"}), 400

    try:
        filepath = os.path.join('./uploads', file.filename)
        file.save(filepath)
        frame = cv2.imread(filepath)
        if frame is None:
            return jsonify({"error": "Unable to read image file!"}), 400

        image = frame.copy()
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)

        points = detection(frame)
        points = util.sort_polygon(list(points))

        for point in points:
            point = np.array(point, dtype=np.int32)
            cv2.polylines(image, [point], True, (0, 255, 0), 2)

        cropped_images = [util.crop_image(frame, x) for x in points]
        cropped_images, angles = classification(cropped_images)

        results, confidences = recognition(cropped_images)

        if isinstance(confidences, list) and all(isinstance(conf, list) for conf in confidences):
            confidences = [float(conf[0]) for conf in confidences]
        else:
            confidences = [float(conf) for conf in confidences]

        for i, result in enumerate(results):
            point = points[i]
            x, y, w, h = cv2.boundingRect(point)
            image = cv2.putText(image, result, (int(x), int(y - 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1, cv2.LINE_AA)

        output_filename = f'processed_{file.filename}'
        output_path = os.path.join('./outputs', output_filename)
        cv2.imwrite(output_path, image)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Search for each recognized medicine name in CSV
        search_results = []
        for medicine_name in results:
            search_data = search_medicine_in_csv(medicine_name)
            if search_data:
                search_results.extend(search_data)

        save_results_to_csv(filepath, results, confidences,search_results)

        return jsonify({
            "results": results,
            "avg_confidence": avg_confidence,
            "output_image": output_filename,
            "medicine_info": search_results if search_results else "No information found for the detected medicines"
        }), 200

    except Exception as e:
        app.logger.error("Error processing image: %s", e)
        return jsonify({"error": "An error occurred while processing the image!"}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        if not os.path.exists('model_results.csv'):
            return jsonify({"message": "No processing history found!"}), 200

        history_data = []
        with open('model_results.csv', mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for index, row in enumerate(reader):  # Use enumerate to get the index
                if len(row) < 2:
                    continue  # Skip invalid rows

                filepath = row[1]
                num_results = int(row[2])
                results = row[3:3 + num_results]
                confidences = row[3 + num_results:]

                # Convert confidences to float
                try:
                    confidences_float = [float(conf) for conf in confidences]
                except ValueError as e:
                    app.logger.error("Error converting confidences: %s", e)
                    confidences_float = ["Invalid confidence"] * len(confidences)  # Handle error

                # Retrieve search results for the recognized medicines
                search_results = []
                for medicine_name in results:
                    search_data = search_medicine_in_csv(medicine_name)
                    if search_data:
                        search_results.extend(search_data)

                # Append the index as id
                history_data.append({
                    "id": index,  # Include the index as an ID
                    "filepath": filepath,
                    "results": results,
                    "confidences": confidences_float,  
                    "search_results": search_results,  
                })

        return jsonify({"history": history_data}), 200

    except Exception as e:
        app.logger.error("Error retrieving history: %s", e)
        return jsonify({"error": "An error occurred while retrieving history!"}), 500

@app.route('/history/<int:id>', methods=['DELETE'])
def delete_history(id):
    try:
        # Read the content from the CSV file
        with open('model_results.csv', mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)

        # Check if the ID is valid
        if id < 0 or id >= len(rows):
            return jsonify({"error": "ID không hợp lệ!"}), 404

        # Delete the corresponding row
        del rows[id]

        # Write the updated content back to the CSV file
        with open('model_results.csv', mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        return jsonify({"message": "Đã xóa lịch sử thành công!"}), 200

    except Exception as e:
        app.logger.error("Error deleting history: %s", e)
        return jsonify({"error": "Đã xảy ra lỗi khi xóa lịch sử!"}), 500

if __name__ == '__main__':
    os.makedirs('./uploads', exist_ok=True)
    os.makedirs('./outputs', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
