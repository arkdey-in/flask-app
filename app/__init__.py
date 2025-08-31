from flask import Flask
from flask_mysqldb import MySQL
import os

mysql = MySQL()

def create_app():
    """Application factory function."""
    app = Flask(__name__)
    
    # Load configuration from config.py
    app.config.from_object('config')
    
    # Initialize extensions
    mysql.init_app(app)
    
    # Ensure upload folder exists
    # This uses the absolute path defined in config.py
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register blueprints within the app context
    with app.app_context():
        from .routes import main as main_blueprint
        app.register_blueprint(main_blueprint)
    
    return app