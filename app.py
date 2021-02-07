from flask import Flask, render_template, jsonify, request
import alignment
import os

app = Flask(__name__)

"""
@app.route("/")
def disclaimer():
    return render_template("disclaimer.html")
"""

#route on webpage load
@app.route("/")
def index():
    return render_template("index.html")    

#route to get result from a selfie submission
@app.route('/getdata', methods=['GET','POST'])
def getdata():

    isthisFile=request.files.get('file')
    print(isthisFile)
    print(isthisFile.filename)
    isthisFile.save("./Aligned_Images/"+"Image_Prediction.jpg")
    
    age, gender = alignment.alignment_func()
    #check for percent if its gtreater then modify else show old values
    #..........

    return jsonify({'gender': gender, 'age' : age})

#route to the live video feed
@app.route('/getLiveVideo', methods=['GET','POST'])
def getLiveVideo():
    return os.system("py detect_age_video.py --face face_detector --age age_detector --gender gender_detector")

if __name__ == '__main__':
    app.run(debug=True)