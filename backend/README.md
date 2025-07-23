# Flask Application

This project is a Flask-based web application designed for educational purposes, focusing on teaching PyTorch through interactive lessons and code execution.

## Project Structure

```
flask-app
├── app.py                # Main application file
├── requirements.txt      # Project dependencies
├── .env                  # Environment variables
├── migrations            # Database migrations
│   └── env.py           # Migration environment configuration
└── README.md             # Project documentation
```

## Requirements

To run this application, you need to install the required dependencies listed in `requirements.txt`. You can do this by running:

```
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the root directory of the project and define the following variables:

```
DATABASE_URL=your_database_url_here
JWT_SECRET_KEY=your_jwt_secret_key_here
FLASK_ENV=development
```

## Running the Application

To start the Flask application, run the following command:

```
python app.py
```

The application will be available at `http://127.0.0.1:5000`.

## API Endpoints

The application provides various API endpoints for:

- User authentication
- Course management
- Code execution
- User progress tracking

Refer to the API documentation for detailed information on each endpoint.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.