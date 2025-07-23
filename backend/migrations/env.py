from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Import models here
from models import User, Course, Lesson, LessonStep, UserProgress, CodeSubmission

# Import routes here
from routes import auth_routes, course_routes, execution_routes, progress_routes

app.register_blueprint(auth_routes)
app.register_blueprint(course_routes)
app.register_blueprint(execution_routes)
app.register_blueprint(progress_routes)

if __name__ == '__main__':
    app.run(debug=True)