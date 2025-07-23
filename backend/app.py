from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
import os

app = Flask(__name__)
CORS(app)

# Load environment variables from .env file
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['ENV'] = os.getenv('FLASK_ENV', 'development')

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
jwt = JWTManager(app)

# Define database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    lessons = db.relationship('Lesson', backref='course', lazy=True)

class Lesson(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    steps = db.relationship('LessonStep', backref='lesson', lazy=True)

class LessonStep(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    instruction = db.Column(db.String(500), nullable=False)
    lesson_id = db.Column(db.Integer, db.ForeignKey('lesson.id'), nullable=False)

class UserProgress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    lesson_id = db.Column(db.Integer, db.ForeignKey('lesson.id'), nullable=False)
    completed = db.Column(db.Boolean, default=False)

class CodeSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    lesson_step_id = db.Column(db.Integer, db.ForeignKey('lesson_step.id'), nullable=False)
    code = db.Column(db.Text, nullable=False)

# Define API routes (example)
@app.route('/api/courses', methods=['GET'])
def get_courses():
    # Implementation for getting courses
    pass

@app.route('/api/lessons/<int:course_id>', methods=['GET'])
def get_lessons(course_id):
    # Implementation for getting lessons for a course
    pass

@app.route('/api/submit-code', methods=['POST'])
def submit_code():
    # Implementation for submitting code
    pass

if __name__ == '__main__':
    app.run(debug=True)