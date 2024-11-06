import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import time
import csv

# Kiểm tra xem có GPU hay không
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hàm tạo dữ liệu giả đầu vào
def generate_fake_input(input_shape):
    return torch.randn(*input_shape, dtype=torch.float32).to(device)

# Hàm đánh giá mô hình
def evaluate_model(model, input_shape, num_samples, batch_size, learning_rate, dropout_rate, epochs):
    model.eval()  # Đặt mô hình ở chế độ đánh giá
    model.to(device)  # Chuyển mô hình vào GPU nếu có
    print("Model loaded successfully.")
    
    print(f"Evaluating model with {num_samples} samples and batch size {batch_size}")
    
    inference_times = []
    y_true = []
    y_pred = []
    
    # Số batch đầy đủ và số mẫu còn lại
    total_batches = num_samples // batch_size
    remaining_samples = num_samples % batch_size

    for _ in range(total_batches):
        fake_input = generate_fake_input(input_shape)
        start_time = time.time()
        with torch.no_grad():  # Tắt tính toán gradient để tăng tốc
            outputs = model(fake_input)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Sinh dữ liệu giả để đánh giá accuracy
        y_true.extend(np.random.randint(0, 2, size=(batch_size,)))
        y_pred_batch = np.random.randint(0, 2, size=(batch_size,))
        y_pred.extend(y_pred_batch)

    # Xử lý các mẫu còn lại (nếu có)
    if remaining_samples > 0:
        remaining_shape = [remaining_samples] + input_shape[1:]
        fake_input = generate_fake_input(remaining_shape)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(fake_input)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        y_true.extend(np.random.randint(0, 2, size=(remaining_samples,)))
        y_pred_batch = np.random.randint(0, 2, size=(remaining_samples,))
        y_pred.extend(y_pred_batch)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Tính toán các chỉ số đánh giá
    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_pred == 1) + 1e-8)
    recall = np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Average Inference Time: {np.mean(inference_times):.4f} seconds")
    print(f"Max Inference Time: {np.max(inference_times):.4f} seconds")
    print(f"Min Inference Time: {np.min(inference_times):.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "inference_times": inference_times,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "epochs": epochs
    }

# Hàm lưu kết quả vào CSV
def save_results_to_csv(results, model_name, filename='model_evaluation_results2.csv'):
    fieldnames = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 
                  'avg_inference_time', 'max_inference_time', 'min_inference_time', 
                  'learning_rate', 'dropout_rate', 'epochs']
    
    # Mở file ở chế độ 'a' để thêm dữ liệu
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Nếu file trống, ghi header vào
        if file.tell() == 0:
            writer.writeheader()
        
        # Ghi kết quả cho mô hình hiện tại
        writer.writerow({
            'model_name': model_name,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'avg_inference_time': np.mean(results['inference_times']),
            'max_inference_time': np.max(results['inference_times']),
            'min_inference_time': np.min(results['inference_times']),
            'learning_rate': results['learning_rate'],
            'dropout_rate': results['dropout_rate'],
            'epochs': results['epochs']
        })

# Siêu tham số cho các mô hình
hyperparameters = {
    'detection': {
        'input_shape': [32, 3, 224, 224],
        'learning_rate': 0.001,
        'dropout_rate': 0.7,
        'epochs': 15
    },
    'classification': {
        'input_shape': [32, 3, 224, 224],
        'learning_rate': 0.001,
        'dropout_rate': 0.7,
        'epochs': 15
    },
    'recognition': {
        'input_shape': [32, 3, 48, 48],
        'learning_rate': 0.001,
        'dropout_rate': 0.7,
        'epochs': 15
    }
}

# Khởi tạo các mô hình ResNet (dùng pretrained model cho các mô hình lớn)
detection_model = models.resnet50(pretrained=True)  # Dùng mô hình đã huấn luyện sẵn
classification_model = models.resnet50(pretrained=True)
recognition_model = models.resnet18(pretrained=True)  # Dùng mô hình nhẹ hơn cho recognition

# Số mẫu và batch size cho đánh giá
num_samples = 100
batch_size = 32

# Đánh giá mô hình phát hiện (Detection)
print("Evaluating detection model")
evaluation_results_detection = evaluate_model(detection_model, hyperparameters['detection']['input_shape'], num_samples, batch_size, hyperparameters['detection']['learning_rate'], hyperparameters['detection']['dropout_rate'], hyperparameters['detection']['epochs'])
save_results_to_csv(evaluation_results_detection, 'Detection Model')

# Đánh giá mô hình phân loại (Classification)
print("Evaluating classification model")
evaluation_results_classification = evaluate_model(classification_model, hyperparameters['classification']['input_shape'], num_samples, batch_size, hyperparameters['classification']['learning_rate'], hyperparameters['classification']['dropout_rate'], hyperparameters['classification']['epochs'])
save_results_to_csv(evaluation_results_classification, 'Classification Model')

# Đánh giá mô hình nhận diện (Recognition)
print("Evaluating recognition model")
evaluation_results_recognition = evaluate_model(recognition_model, hyperparameters['recognition']['input_shape'], num_samples, batch_size, hyperparameters['recognition']['learning_rate'], hyperparameters['recognition']['dropout_rate'], hyperparameters['recognition']['epochs'])
save_results_to_csv(evaluation_results_recognition, 'Recognition Model')
