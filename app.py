from flask import Flask
app = Flask(__name__)  # Note the correct use of '__name__' (not '_name__')

@app.route('/')
def hello_world():
    return 'Hello Jigme'

if __name__ == '__main__':
    app.run()  # Correct method to start the Flask server
