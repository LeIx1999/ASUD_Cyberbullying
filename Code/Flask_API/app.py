from flask import Flask, jsonify, request, render_template
from sentence_transformers import SentenceTransformer
from joblib import load

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
    tweet = clean_tweet(input_json["text"])
    tweet = [model.encode(tweet)]

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

# new function to clean tweets
def clean_tweet(tweet: str):
    replace_dict = {"*": " ",
                   "*amp;": "und",
                   '&lt;3': '<3',
                   '&lt;': 'kleiner als',
                   '&rt;': 'groesser als',
                    "#": "",
                    "<U+": "",
                    "|LBR|": ""
                   }
    tweet_list = []
    for word in tweet.split(" "):
        for key, value in replace_dict.items():
            word = word.replace(key, value)
        if not "@" in word and not "#" in word:
            tweet_list.append(word)
    return " ".join(tweet_list)



