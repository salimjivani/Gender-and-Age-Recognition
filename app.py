from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.route("/")
def disclaimer():
    return render_template("disclaimer.html")

@app.route("/index")
def index():
    return render_template("index.html")    

@app.route('/getdata', methods=['GET','POST'])
def getdata():

    isthisFile=request.files.get('file')
    print(isthisFile)
    print(isthisFile.filename)
    isthisFile.save("./Aligned_Images/"+isthisFile.filename)

    return jsonify({'gender': "M", 'age' : "(25,32)"})

if __name__ == '__main__':
    app.run(debug=True)
