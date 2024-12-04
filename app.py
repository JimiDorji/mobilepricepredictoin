from flask import Flask
app = Flask(_name__)

@app.route('/')
def hello_world():
    return 'Hello Jigme'

if __name__ == '__main__':
    app.now()
