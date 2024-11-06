import numpy as np
import time
import csv
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Hàm tạo dữ liệu giả cho testing
def generate_fake_input(input_shape):
    return np.random.rand(*input_shape).astype(np.float64)

# Đánh giá mô hình
def evaluate_model(model_name, model, input_shape, num_samples, batch_size, learning_rate, dropout_rate, epochs):
    inference_times = []
    y_true = []
    y_pred = []

    total_batches = num_samples // batch_size
    remaining_samples = num_samples % batch_size

    for _ in range(total_batches):
        fake_input = generate_fake_input(input_shape)
        start_time = time.time()
        output = model(fake_input)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        y_true.extend(np.random.randint(0, 2, size=(batch_size,)))
        y_pred_batch = np.random.randint(0, 2, size=(batch_size,))  # Simulate predictions for the batch
        y_pred.extend(y_pred_batch)

    if remaining_samples > 0:
        remaining_shape = [remaining_samples] + input_shape[1:]
        fake_input = generate_fake_input(remaining_shape)
        start_time = time.time()
        output = model(fake_input)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        y_true.extend(np.random.randint(0, 2, size=(remaining_samples,)))
        y_pred_batch = np.random.randint(0, 2, size=(remaining_samples,))  # Handle the remaining samples
        y_pred.extend(y_pred_batch)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure that y_true and y_pred have the same shape
    assert y_true.shape == y_pred.shape, f"Shape mismatch: y_true shape = {y_true.shape}, y_pred shape = {y_pred.shape}"

    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_pred == 1) + 1e-8)
    recall = np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Model {model_name}:")
    print(f"Avg Inference Time: {np.mean(inference_times):.4f} sec")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
    print(f"Batch Size: {batch_size}, Learning Rate: {learning_rate}, Dropout Rate: {dropout_rate}, Epochs: {epochs}")

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "avg_inference_time": np.mean(inference_times),
        "max_inference_time": np.max(inference_times),
        "min_inference_time": np.min(inference_times),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "epochs": epochs
    }

# Lưu kết quả vào CSV
def save_results_to_csv(results, filename='model_evaluation_results.csv'):
    fieldnames = [
        'model_name', 'accuracy', 'precision', 'recall', 'f1_score', 
        'avg_inference_time', 'max_inference_time', 'min_inference_time',
        'batch_size', 'learning_rate', 'dropout_rate', 'epochs'
    ]
    
    # Mở file ở chế độ 'a' để thêm dữ liệu
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Nếu file trống, ghi header vào
        if file.tell() == 0:
            writer.writeheader()
        
        # Ghi kết quả cho mô hình hiện tại
        writer.writerow(results)

# Hàm Fine-tuning MobileNetV2
def fine_tune_model(base_model, input_shape, learning_rate, dropout_rate):
    base_model.trainable = False  # Lớp base không được huấn luyện ban đầu
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Fine-tuning với các mô hình khác như ResNet50, EfficientNetB0
def fine_tune_resnet50(input_shape, learning_rate, dropout_rate):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    return fine_tune_model(base_model, input_shape, learning_rate, dropout_rate)

def fine_tune_efficientnetb0(input_shape, learning_rate, dropout_rate):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    return fine_tune_model(base_model, input_shape, learning_rate, dropout_rate)

# Fine-tuning MobileNetV2 cho Recognition
def fine_tune_mobilenetv2_recognition(input_shape, learning_rate, dropout_rate):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    return fine_tune_model(base_model, input_shape, learning_rate, dropout_rate)

# Tăng cường dữ liệu (Data Augmentation)
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Cấu hình siêu tham số
hyperparameters = {
    'detection': {'input_shape': [1, 224, 224, 3]},
    'recognition': {'input_shape': [1, 48, 48, 3]},
    'classification': {'input_shape': [1, 224, 224, 3]},
    'batch_size': 64,
    'learning_rate': 0.01,
    'dropout_rate': 0.1,
    'epochs': 15
}

# Đưa mô hình MobileNetV2 vào fine-tuning
base_model_mobilenetv2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_mobilenetv2 = fine_tune_model(base_model_mobilenetv2, (224, 224, 3), hyperparameters['learning_rate'], hyperparameters['dropout_rate'])

# Đưa mô hình ResNet50 vào fine-tuning
model_resnet50 = fine_tune_resnet50((224, 224, 3), hyperparameters['learning_rate'], hyperparameters['dropout_rate'])

# Đưa mô hình EfficientNetB0 vào fine-tuning
model_efficientnetb0 = fine_tune_efficientnetb0((224, 224, 3), hyperparameters['learning_rate'], hyperparameters['dropout_rate'])

# Đưa mô hình nhận dạng MobileNetV2 cho Recognition vào fine-tuning
model_recognition = fine_tune_mobilenetv2_recognition((48, 48, 3), hyperparameters['learning_rate'], hyperparameters['dropout_rate'])

# Sử dụng EarlyStopping để tránh overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Đánh giá mô hình và lưu kết quả
num_samples = 200
batch_size = hyperparameters['batch_size']

# Đánh giá mô hình nhận dạng MobileNetV2
print("Evaluating MobileNetV2 Classification Model...")
classification_results_mobilenetv2 = evaluate_model("MobileNetV2 Classification Model", model_mobilenetv2, hyperparameters['classification']['input_shape'], num_samples, batch_size, hyperparameters['learning_rate'], hyperparameters['dropout_rate'], hyperparameters['epochs'])
save_results_to_csv(classification_results_mobilenetv2)

# Đánh giá mô hình nhận dạng ResNet50
print("Evaluating ResNet50 Classification Model...")
classification_results_resnet50 = evaluate_model("ResNet50 Classification Model", model_resnet50, hyperparameters['classification']['input_shape'], num_samples, batch_size, hyperparameters['learning_rate'], hyperparameters['dropout_rate'], hyperparameters['epochs'])
save_results_to_csv(classification_results_resnet50)

# Đánh giá mô hình nhận dạng EfficientNetB0
print("Evaluating EfficientNetB0 Classification Model...")
classification_results_efficientnetb0 = evaluate_model("EfficientNetB0 Classification Model", model_efficientnetb0, hyperparameters['classification']['input_shape'], num_samples, batch_size, hyperparameters['learning_rate'], hyperparameters['dropout_rate'], hyperparameters['epochs'])
save_results_to_csv(classification_results_efficientnetb0)

# Đánh giá mô hình nhận dạng MobileNetV2 cho Recognition
print("Evaluating MobileNetV2 Recognition Model...")
recognition_results = evaluate_model("MobileNetV2 Recognition Model", model_recognition, hyperparameters['recognition']['input_shape'], num_samples, batch_size, hyperparameters['learning_rate'], hyperparameters['dropout_rate'], hyperparameters['epochs'])
save_results_to_csv(recognition_results)