from flask import Flask, jsonify, request, render_template
from sentence_transformers import SentenceTransformer
from joblib import load
from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__)

@app.route('/')
def show_information():
    return render_template("about.html")

@app.route('/cyberbullying', methods = ["POST"])
def make_predictions():
    # get the post body
    input_json = request.get_json(force=True)

    # load pretrained transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # create tweet embeddings
    tweet = [model.encode(input_json["text"])]

    # load random forest model
    rf_model = load('model.joblib')

    # make predictions
    preds = rf_model.predict(tweet)

    # decode predictions
    enc_dict = {0: "ABUSE",
                1: "INSULT",
                2: "OTHER",
                3: "PROFANITY"}
    preds = enc_dict[preds[0]]
    dictToReturn = {'text': preds}
    return jsonify(dictToReturn)



