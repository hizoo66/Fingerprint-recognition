from flask import Flask, render_template

app = Flask("Finger print_recognization")

@app.route("/")
def hello():
    return render_template("index.html")


app.run("")