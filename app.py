from flask import Flask

app = Flask("Finger print_recognization")

@app.route("/")
def hello():
    return "Hello World!"

app.run("")