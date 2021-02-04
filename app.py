from flask import Flask, render_template, jsonify, request
import detect_age

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/getdata', methods=['POST'])
def getdata():
    
    exec(open('alignment.py').read(), globals(), globals())
    age = detect_age.age_value
    gender = detect_age.gender_value
    return jsonify({'gender': gender, 'age' : age})

if __name__ == '__main__':
    app.run(debug=True)
