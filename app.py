from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/getdata', methods=['POST'])
def getdata():
    return jsonify({'gender': "M", 'age' : "(25,32)"})

if __name__ == '__main__':
    app.run(debug=True)
