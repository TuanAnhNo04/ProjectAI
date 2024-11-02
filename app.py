import time
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Tạo dữ liệu đầu vào giả để kiểm thử mô hình
def generate_fake_input(input_shape):
    return np.random.rand(*input_shape).astype(np.float32)

# Kiểm tra thông tin đầu vào và đầu ra của mô hình
def check_model_io(model_path):
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    print(f"Checking model: {model_path}")
    
    # In thông tin đầu vào
    for input_meta in session.get_inputs():
        print(f"Input name: {input_meta.name}")
        print(f"Input shape: {input_meta.shape}")
        print(f"Input type: {input_meta.type}")

    # In thông tin đầu ra
    for output_meta in session.get_outputs():
        print(f"Output name: {output_meta.name}")
        print(f"Output shape: {output_meta.shape}")
        print(f"Output type: {output_meta.type}")

# Đánh giá mô hình
def evaluate_model(model_path, input_shape, num_samples=100):
    # Khởi tạo session với CPUExecutionProvider
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    inference_times = []
    y_true = []  # Giá trị thực
    y_pred = []  # Giá trị dự đoán
    
    for _ in range(num_samples):
        # Tạo dữ liệu đầu vào
        fake_input = generate_fake_input(input_shape)

        # Bắt đầu đo thời gian suy luận
        start_time = time.time()

        # Thực hiện suy luận
        output = session.run(None, {session.get_inputs()[0].name: fake_input})

        # Giả định y_pred là một nhãn dự đoán
        # Điều chỉnh theo mô hình thực tế để lấy y_pred
        y_pred.append(np.argmax(output[0], axis=1))  # Chỉ dành cho classification
        y_true.append(np.random.randint(0, output[0].shape[1], size=fake_input.shape[0]))  # Giả định nhãn thật

        end_time = time.time()
        inference_times.append(end_time - start_time)

    # Tính toán thống kê thời gian suy luận
    avg_time = np.mean(inference_times)
    max_time = np.max(inference_times)
    min_time = np.min(inference_times)

    # Tính toán các chỉ số đánh giá
    y_true_flat = np.concatenate(y_true)
    y_pred_flat = np.concatenate(y_pred)

    accuracy = np.mean(y_true_flat == y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)

    print(f"Average Inference Time: {avg_time:.4f} seconds")
    print(f"Max Inference Time: {max_time:.4f} seconds")
    print(f"Min Inference Time: {min_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return avg_time, max_time, min_time, accuracy, precision, recall, f1

# Đánh giá từng mô hình
results = []

# Kiểm tra mô hình Detection
check_model_io('./weights/detection.onnx')
try:
    print("\nEvaluating Detection Model:")
    results.append(evaluate_model('./weights/detection.onnx', input_shape=(1, 3, 640, 640)))
except Exception as e:
    print(f"Error evaluating Detection model: {e}")

# Kiểm tra mô hình Recognition
check_model_io('./weights/recognition.onnx')
try:
    print("\nEvaluating Recognition Model:")
    results.append(evaluate_model('./weights/recognition.onnx', input_shape=(1, 3, 48, 48)))  # Điều chỉnh theo yêu cầu
except Exception as e:
    print(f"Error evaluating Recognition model: {e}")

# Kiểm tra mô hình Classification
check_model_io('./weights/classification.onnx')
try:
    print("\nEvaluating Classification Model:")
    results.append(evaluate_model('./weights/classification.onnx', input_shape=(1, 3, 224, 224)))
except Exception as e:
    print(f"Error evaluating Classification model: {e}")

# Lưu kết quả vào file CSV
columns = ['Model', 'Average Inference Time', 'Max Inference Time', 'Min Inference Time', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
df_results = pd.DataFrame(results, columns=columns[1:])
df_results.insert(0, 'Model', ['Detection', 'Recognition', 'Classification'])
df_results.to_csv('model_evaluation_results.csv', index=False)

print("\nResults have been saved to 'model_evaluation_results.csv'.")
