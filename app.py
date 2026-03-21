from flask import Flask,render_template,request
from src.predict import predict_review

app=Flask(__name__)

@app.route("/",methods=["GET","POST"])
def home():
    result=None
    threshold=None

    if request.method=="POST":
        review=request.form["review"]
        rating=int(request.form["rating"])

        result,threshold=predict_review(review,rating)

    return render_template("index.html",result=result,threshold=threshold)

if __name__=="__main__":
    app.run(debug=True)