import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import time
import csv

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to generate fake input data
def generate_fake_input(input_shape):
    return torch.randn(*input_shape, dtype=torch.float32).to(device)

# Model evaluation function
def evaluate_model(model, input_shape, num_samples, batch_size, learning_rate, dropout_rate, epochs):
    model.eval()  # Set model to evaluation mode
    model.to(device)  # Move model to GPU if available
    print("Model loaded successfully.")
    
    print(f"Evaluating model with {num_samples} samples and batch size {batch_size}")
    
    inference_times = []
    y_true = []
    y_pred = []
    
    # Full batches and remaining samples
    total_batches = num_samples // batch_size
    remaining_samples = num_samples % batch_size

    for _ in range(total_batches):
        fake_input = generate_fake_input(input_shape)
        start_time = time.time()
        with torch.no_grad():  # Disable gradient calculation for faster evaluation
            outputs = model(fake_input)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Generate random labels for fake accuracy
        y_true.extend(np.random.randint(0, 2, size=(batch_size,)))
        y_pred_batch = np.random.randint(0, 2, size=(batch_size,))
        y_pred.extend(y_pred_batch)

    # Process remaining samples if any
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

    # Calculate metrics
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

# Function to save results to CSV
def save_results_to_csv(results, model_name, filename='model_evaluation_results3.csv'):
    fieldnames = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 
                  'avg_inference_time', 'max_inference_time', 'min_inference_time', 
                  'learning_rate', 'dropout_rate', 'epochs']
    
    # Open file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write header if file is empty
        if file.tell() == 0:
            writer.writeheader()
        
        # Write model results
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

# Hyperparameters for each model
hyperparameters = {
    'detection': {
        'input_shape': [32, 3, 224, 224],
        'learning_rate': 0.001,
        'dropout_rate': 0.5,
        'epochs': 10
    },
    'classification': {
        'input_shape': [32, 3, 224, 224],
        'learning_rate': 0.001,
        'dropout_rate': 0.5,
        'epochs': 10
    },
    'recognition': {
        'input_shape': [32, 3, 48, 48],
        'learning_rate': 0.001,
        'dropout_rate': 0.5,
        'epochs': 10
    }
}

# Initialize R-CNN and SSD models
detection_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
ssd_model = models.detection.ssd300_vgg16(pretrained=True)  # Change if SSD model is pre-trained

# Recognition model (ResNet-18 as a lightweight alternative)
recognition_model = models.resnet18(pretrained=True)

# Number of samples and batch size
num_samples = 100
batch_size = 32

# Evaluate R-CNN model
print("Evaluating R-CNN model")
evaluation_results_rcnn = evaluate_model(
    detection_model, 
    hyperparameters['detection']['input_shape'], 
    num_samples, 
    batch_size, 
    hyperparameters['detection']['learning_rate'], 
    hyperparameters['detection']['dropout_rate'], 
    hyperparameters['detection']['epochs']
)
save_results_to_csv(evaluation_results_rcnn, 'R-CNN Model', filename='model_evaluation_results3.csv')

# Evaluate SSD model
print("Evaluating SSD model")
evaluation_results_ssd = evaluate_model(
    ssd_model, 
    hyperparameters['classification']['input_shape'], 
    num_samples, 
    batch_size, 
    hyperparameters['classification']['learning_rate'], 
    hyperparameters['classification']['dropout_rate'], 
    hyperparameters['classification']['epochs']
)
save_results_to_csv(evaluation_results_ssd, 'SSD Model', filename='model_evaluation_results3.csv')

# Evaluate recognition model
print("Evaluating recognition model")
evaluation_results_recognition = evaluate_model(
    recognition_model, 
    hyperparameters['recognition']['input_shape'], 
    num_samples, 
    batch_size, 
    hyperparameters['recognition']['learning_rate'], 
    hyperparameters['recognition']['dropout_rate'], 
    hyperparameters['recognition']['epochs']
)
save_results_to_csv(evaluation_results_recognition, 'Recognition Model', filename='model_evaluation_results3.csv')