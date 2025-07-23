from flask import Flask, jsonify, request
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for Vue.js frontend

# Sample learning modules data
learning_modules = {
    "pytorch": {
        "title": "PyTorch Fundamentals",
        "description": "Learn the basics of PyTorch for deep learning",
        "lessons": [
            {
                "id": 1,
                "title": "Tensors and Basic Operations",
                "content": "PyTorch tensors are the fundamental building blocks for deep learning.",
                "code_example": """import torch

# Create tensors
x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([5, 6, 7, 8])

# Basic operations
addition = x + y
multiplication = x * y
dot_product = torch.dot(x, y)

print(f"Addition: {addition}")
print(f"Multiplication: {multiplication}")
print(f"Dot Product: {dot_product}")""",
                "difficulty": "Beginner"
            },
            {
                "id": 2,
                "title": "Neural Networks with nn.Module",
                "content": "Build your first neural network using PyTorch's nn.Module.",
                "code_example": """import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Create model
model = SimpleNet(10, 5, 1)
print(model)""",
                "difficulty": "Intermediate"
            }
        ]
    },
    "tensorflow": {
        "title": "TensorFlow & Keras",
        "description": "Master TensorFlow and Keras for machine learning",
        "lessons": [
            {
                "id": 1,
                "title": "Getting Started with TensorFlow",
                "content": "Introduction to TensorFlow tensors and basic operations.",
                "code_example": """import tensorflow as tf

# Create tensors
a = tf.constant([1, 2, 3, 4])
b = tf.constant([5, 6, 7, 8])

# Operations
addition = tf.add(a, b)
multiplication = tf.multiply(a, b)
dot_product = tf.tensordot(a, b, axes=1)

print(f"Addition: {addition}")
print(f"Multiplication: {multiplication}")
print(f"Dot Product: {dot_product}")""",
                "difficulty": "Beginner"
            },
            {
                "id": 2,
                "title": "Building Models with Keras",
                "content": "Create neural networks using Keras Sequential API.",
                "code_example": """import tensorflow as tf
from tensorflow import keras

# Create a simple sequential model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()""",
                "difficulty": "Intermediate"
            }
        ]
    },
    "transformers": {
        "title": "Transformers & NLP",
        "description": "Explore transformer models for natural language processing",
        "lessons": [
            {
                "id": 1,
                "title": "Introduction to Transformers",
                "content": "Understanding the transformer architecture and attention mechanism.",
                "code_example": """from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tokenize input text
text = "Hello, world! This is a transformer example."
inputs = tokenizer(text, return_tensors="pt")

# Get model outputs
with torch.no_grad():
    outputs = model(**inputs)

print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Output shape: {outputs.last_hidden_state.shape}")""",
                "difficulty": "Advanced"
            },
            {
                "id": 2,
                "title": "Fine-tuning for Classification",
                "content": "Fine-tune a transformer model for text classification.",
                "code_example": """from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

# Load model for classification
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2
)

# Example training setup
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

print("Model ready for fine-tuning!")""",
                "difficulty": "Advanced"
            }
        ]
    }
}

# User progress tracking (in-memory for simplicity)
user_progress = {}

@app.route('/api/modules', methods=['GET'])
def get_modules():
    """Get all learning modules"""
    return jsonify(learning_modules)

@app.route('/api/modules/<module_name>', methods=['GET'])
def get_module(module_name):
    """Get specific module details"""
    if module_name in learning_modules:
        return jsonify(learning_modules[module_name])
    return jsonify({"error": "Module not found"}), 404

@app.route('/api/modules/<module_name>/lessons/<int:lesson_id>', methods=['GET'])
def get_lesson(module_name, lesson_id):
    """Get specific lesson details"""
    if module_name in learning_modules:
        lessons = learning_modules[module_name]['lessons']
        lesson = next((l for l in lessons if l['id'] == lesson_id), None)
        if lesson:
            return jsonify(lesson)
    return jsonify({"error": "Lesson not found"}), 404

@app.route('/api/progress', methods=['POST'])
def update_progress():
    """Update user progress"""
    data = request.json
    user_id = data.get('user_id', 'default')
    module = data.get('module')
    lesson_id = data.get('lesson_id')
    completed = data.get('completed', False)
    
    if user_id not in user_progress:
        user_progress[user_id] = {}
    
    if module not in user_progress[user_id]:
        user_progress[user_id][module] = {}
    
    user_progress[user_id][module][lesson_id] = completed
    
    return jsonify({"message": "Progress updated successfully"})

@app.route('/api/progress/<user_id>', methods=['GET'])
def get_progress(user_id):
    """Get user progress"""
    return jsonify(user_progress.get(user_id, {}))

@app.route('/api/run-code', methods=['POST'])
def run_code():
    """Simulate code execution (for demo purposes)"""
    data = request.json
    code = data.get('code', '')
    
    # In a real implementation, you'd want to use a secure code execution environment
    # For this demo, we'll just return a mock response
    return jsonify({
        "output": "Code executed successfully!\n(This is a demo - actual execution would require a secure sandbox)",
        "success": True
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)