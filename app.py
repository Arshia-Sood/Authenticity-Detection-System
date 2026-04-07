from flask import Flask,render_template,request,redirect
from src.predict import predict_review
from flask_sqlalchemy import SQLAlchemy
import os

app=Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
db=SQLAlchemy(app)

class Review(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    review_text=db.Column(db.Text)
    rating=db.Column(db.Integer)
    score=db.Column(db.Float)
    risk=db.Column(db.Float)
    decision=db.Column(db.String(20))

@app.before_first_request
def create_tables():
    db.create_all()

@app.route("/",methods=["GET","POST"])
def home():
    result=None
    threshold=None

    if request.method=="POST":
        review=request.form["review"]
        rating=int(request.form["rating"])

        result,threshold=predict_review(review,rating)

        new_review=Review(
            review_text=review,
            rating=rating,
            score=result["score"],
            risk=result["risk"],
            decision=result["decision"]
        )

        db.session.add(new_review)
        db.session.commit()

    return render_template("index.html",result=result,threshold=threshold)

@app.route("/history")
def history():
    reviews = Review.query.order_by(Review.id.desc()).all()
    return render_template("history.html", reviews=reviews)

@app.route("/clear_history", methods=["POST"])
def clear_history():
    Review.query.delete()
    db.session.commit()
    return redirect("/history")

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
    