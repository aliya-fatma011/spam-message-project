from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/', methods=['GET','POST'])
def home():
    result = ""
    if request.method == 'POST':
        msg = request.form['message']
        vec = vectorizer.transform([msg])
        pred = model.predict(vec)
        result = "🚨 SPAM MESSAGE" if pred[0]==1 else "✅ NOT SPAM"
    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)
