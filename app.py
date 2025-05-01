from flask import Flask
from blueprints.index import bp as index_bp
from blueprints.functionsPage import bp as functions_bp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fx258369'

app.register_blueprint(index_bp)
app.register_blueprint(functions_bp)

if __name__ == '__main__':
    app.run()