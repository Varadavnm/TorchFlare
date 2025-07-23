from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import traceback
import sys
from io import StringIO
import contextlib

app = Flask(__name__)
CORS(app)

# Sample lesson data structure
LESSONS = {
    "1": {
        "id": "1",
        "title": "Introduction to PyTorch Tensors",
        "description": "Learn the basics of PyTorch tensors and operations",
        "level": "beginner",
        "steps": [
            {
                "id": 1,
                "instruction": "Import PyTorch and create your first tensor",
                "hint": "Use 'import torch' and torch.tensor() to create a tensor",
                "starter_code": "# Import PyTorch\n# TODO: Import torch\n\n# Create a simple tensor\n# TODO: Create a tensor with values [1, 2, 3, 4]",
                "solution": "import torch\n\n# Create a simple tensor\ntensor = torch.tensor([1, 2, 3, 4])\nprint(tensor)",
                "expected_output": "tensor([1, 2, 3, 4])"
            },
            {
                "id": 2,
                "instruction": "Create tensors with different shapes and data types",
                "hint": "Use torch.zeros(), torch.ones(), and specify dtype parameter",
                "starter_code": "import torch\n\n# TODO: Create a 3x3 tensor of zeros\n# TODO: Create a 2x4 tensor of ones with float32 type\n# TODO: Create a random tensor of shape (2, 3)",
                "solution": "import torch\n\n# Create a 3x3 tensor of zeros\nzeros_tensor = torch.zeros(3, 3)\nprint('Zeros tensor:', zeros_tensor)\n\n# Create a 2x4 tensor of ones with float32 type\nones_tensor = torch.ones(2, 4, dtype=torch.float32)\nprint('Ones tensor:', ones_tensor)\n\n# Create a random tensor of shape (2, 3)\nrandom_tensor = torch.randn(2, 3)\nprint('Random tensor:', random_tensor)",
                "expected_output": "Various tensor outputs with specified shapes"
            }
        ]
    },
    "2": {
        "id": "2",
        "title": "Tensor Operations and Broadcasting",
        "description": "Learn tensor arithmetic and broadcasting rules",
        "level": "beginner",
        "steps": [
            {
                "id": 1,
                "instruction": "Perform basic tensor arithmetic operations",
                "hint": "Use +, -, *, / operators or torch.add(), torch.sub(), etc.",
                "starter_code": "import torch\n\n# Create two tensors\na = torch.tensor([1, 2, 3])\nb = torch.tensor([4, 5, 6])\n\n# TODO: Add the tensors\n# TODO: Multiply the tensors\n# TODO: Compute element-wise division",
                "solution": "import torch\n\n# Create two tensors\na = torch.tensor([1, 2, 3])\nb = torch.tensor([4, 5, 6])\n\n# Add the tensors\naddition = a + b\nprint('Addition:', addition)\n\n# Multiply the tensors\nmultiplication = a * b\nprint('Multiplication:', multiplication)\n\n# Compute element-wise division\ndivision = b / a\nprint('Division:', division)",
                "expected_output": "Addition: tensor([5, 7, 9])\nMultiplication: tensor([4, 10, 18])\nDivision: tensor([4.0000, 2.5000, 2.0000])"
            }
        ]
    },
    "3": {
        "id": "3",
        "title": "Building Your First Neural Network",
        "description": "Create a simple neural network using torch.nn",
        "level": "intermediate",
        "steps": [
            {
                "id": 1,
                "instruction": "Import necessary modules and create a simple linear layer",
                "hint": "Use torch.nn.Linear for linear layers",
                "starter_code": "import torch\nimport torch.nn as nn\n\n# TODO: Create a linear layer that takes 10 inputs and produces 5 outputs\n# TODO: Create some sample input data (batch_size=3, features=10)\n# TODO: Pass the input through the linear layer",
                "solution": "import torch\nimport torch.nn as nn\n\n# Create a linear layer that takes 10 inputs and produces 5 outputs\nlinear = nn.Linear(10, 5)\n\n# Create some sample input data (batch_size=3, features=10)\ninput_data = torch.randn(3, 10)\n\n# Pass the input through the linear layer\noutput = linear(input_data)\nprint('Input shape:', input_data.shape)\nprint('Output shape:', output.shape)\nprint('Output:', output)",
                "expected_output": "Input shape: torch.Size([3, 10])\nOutput shape: torch.Size([3, 5])\nOutput: tensor([[...], [...], [...]])"
            }
        ]
    }
}

@app.route('/api/lessons', methods=['GET'])
def get_lessons():
    """Get all available lessons"""
    lessons_list = []
    for lesson_id, lesson in LESSONS.items():
        lessons_list.append({
            'id': lesson['id'],
            'title': lesson['title'],
            'description': lesson['description'],
            'level': lesson['level'],
            'steps_count': len(lesson['steps'])
        })
    return jsonify(lessons_list)

@app.route('/api/lessons/<lesson_id>', methods=['GET'])
def get_lesson(lesson_id):
    """Get a specific lesson with all its steps"""
    if lesson_id not in LESSONS:
        return jsonify({'error': 'Lesson not found'}), 404
    return jsonify(LESSONS[lesson_id])

@app.route('/api/lessons/<lesson_id>/steps/<int:step_id>', methods=['GET'])
def get_step(lesson_id, step_id):
    """Get a specific step from a lesson"""
    if lesson_id not in LESSONS:
        return jsonify({'error': 'Lesson not found'}), 404
    
    lesson = LESSONS[lesson_id]
    step = next((s for s in lesson['steps'] if s['id'] == step_id), None)
    
    if not step:
        return jsonify({'error': 'Step not found'}), 404
    
    return jsonify(step)

@app.route('/api/execute', methods=['POST'])
def execute_code():
    """Execute user's PyTorch code and return results"""
    try:
        data = request.json
        code = data.get('code', '')
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # Create a namespace for code execution
        namespace = {}
        
        try:
            # Execute the code
            exec(code, namespace)
            output = captured_output.getvalue()
            
            return jsonify({
                'success': True,
                'output': output,
                'error': None
            })
            
        except Exception as e:
            error_msg = str(e)
            traceback_msg = traceback.format_exc()
            
            return jsonify({
                'success': False,
                'output': captured_output.getvalue(),
                'error': error_msg,
                'traceback': traceback_msg
            })
            
        finally:
            sys.stdout = old_stdout
            
    except Exception as e:
        return jsonify({
            'success': False,
            'output': '',
            'error': f'Server error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/check-solution/<lesson_id>/<int:step_id>', methods=['POST'])
def check_solution(lesson_id, step_id):
    """Check if the user's code matches the expected solution"""
    try:
        data = request.json
        user_code = data.get('code', '')
        
        if lesson_id not in LESSONS:
            return jsonify({'error': 'Lesson not found'}), 404
        
        lesson = LESSONS[lesson_id]
        step = next((s for s in lesson['steps'] if s['id'] == step_id), None)
        
        if not step:
            return jsonify({'error': 'Step not found'}), 404
        
        # Execute user code
        old_stdout = sys.stdout
        sys.stdout = user_output = StringIO()
        
        try:
            exec(user_code)
            user_result = user_output.getvalue().strip()
        except Exception as e:
            return jsonify({
                'correct': False,
                'message': f'Code execution error: {str(e)}',
                'expected': step.get('expected_output', ''),
                'actual': ''
            })
        finally:
            sys.stdout = old_stdout
        
        # Simple check - in a real implementation, you'd want more sophisticated checking
        expected = step.get('expected_output', '').strip()
        
        # For this demo, we'll consider it correct if the code runs without error
        # In practice, you'd want to check outputs more carefully
        is_correct = len(user_result) > 0 or 'torch' in user_code
        
        return jsonify({
            'correct': is_correct,
            'message': 'Great job!' if is_correct else 'Try again! Check the hints.',
            'expected': expected,
            'actual': user_result
        })
        
    except Exception as e:
        return jsonify({
            'correct': False,
            'message': f'Server error: {str(e)}',
            'expected': '',
            'actual': ''
        }), 500

@app.route('/api/hint/<lesson_id>/<int:step_id>', methods=['GET'])
def get_hint(lesson_id, step_id):
    """Get hint for a specific step"""
    if lesson_id not in LESSONS:
        return jsonify({'error': 'Lesson not found'}), 404
    
    lesson = LESSONS[lesson_id]
    step = next((s for s in lesson['steps'] if s['id'] == step_id), None)
    
    if not step:
        return jsonify({'error': 'Step not found'}), 404
    
    return jsonify({'hint': step.get('hint', 'No hint available')})

@app.route('/api/solution/<lesson_id>/<int:step_id>', methods=['GET'])
def get_solution(lesson_id, step_id):
    """Get the solution for a specific step"""
    if lesson_id not in LESSONS:
        return jsonify({'error': 'Lesson not found'}), 404
    
    lesson = LESSONS[lesson_id]
    step = next((s for s in lesson['steps'] if s['id'] == step_id), None)
    
    if not step:
        return jsonify({'error': 'Step not found'}), 404
    
    return jsonify({'solution': step.get('solution', 'No solution available')})

if __name__ == '__main__':
    app.run(debug=True, port=5000)