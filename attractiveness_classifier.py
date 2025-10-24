from gensim.models import KeyedVectors
import gensim.downloader as api
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import json
from datetime import datetime

# Ensure required NLTK data is available
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")


class AttractivenessClassifier:
    def __init__(self):
        print("🔤 Loading word embeddings...")
        self.word_vectors = self._load_embeddings()

        # Initialize and train the classifier
        self._initialize_classifier()

        # Load stop words
        self.stop_words = set(stopwords.words("english"))
        print("✅ Classifier ready!")

    def _load_embeddings(self):
        """Try to load local, fallback to online Gensim source, then fallback to smaller model"""
        local_path = "/Users/n.chinlue/RealMatch/models/GoogleNews-vectors-negative300.bin.gz"

        try:
            if os.path.exists(local_path):
                print(f"📁 Found local model: {local_path}")
                return KeyedVectors.load_word2vec_format(local_path, binary=True)

            print("🌐 Local model not found — downloading via Gensim API...")
            return api.load("word2vec-google-news-300")

        except Exception as e:
            print(f"⚠️ Could not load GoogleNews model ({e})")
            print("➡️ Falling back to smaller model: glove-wiki-gigaword-100")
            return api.load("glove-wiki-gigaword-100")

    def _initialize_classifier(self):
        """Initialize and train the RandomForest with predefined word categories"""
        attractive_words = [
            "confident", "ambitious", "successful", "intelligent", "passionate",
            "creative", "determined", "motivated", "authentic", "accomplished",
            "innovative", "skilled", "experienced", "qualified", "professional",
            "dedicated", "reliable", "trustworthy", "ethical", "committed",
            "leader", "expert", "specialist", "proficient", "knowledgeable"
        ]

        non_attractive_words = [
            "unemployed", "inexperienced", "unskilled", "unreliable", "incompetent",
            "lazy", "unmotivated", "careless", "unprofessional", "irresponsible",
            "unqualified", "amateur", "mediocre", "average", "basic",
            "struggling", "failing", "confused", "uncertain", "hesitant"
        ]

        X, y = [], []

        for word in attractive_words:
            if word in self.word_vectors:
                X.append(self.word_vectors[word])
                y.append(1)

        for word in non_attractive_words:
            if word in self.word_vectors:
                X.append(self.word_vectors[word])
                y.append(0)

        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X, y)

    def get_word_vector(self, word):
        """Get the vector for a word if available"""
        try:
            return self.word_vectors[word.lower()]
        except KeyError:
            return None

    def classify_word(self, word):
        """Predict whether a word is attractive or not"""
        vector = self.get_word_vector(word)
        if vector is not None:
            pred = self.classifier.predict([vector])[0]
            prob = self.classifier.predict_proba([vector])[0][1]
            return pred, prob
        return None, None

    def analyze_text(self, text):
        """Analyze a block of text for linguistic attractiveness"""
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)

        results = {
            "attractive_words": [],
            "non_attractive_words": [],
            "summary": {
                "total_analyzed": 0,
                "attractive_count": 0,
                "non_attractive_count": 0,
                "attractiveness_score": 0.0,
            },
        }

        analyzed_count = 0
        attractive_score_sum = 0

        for word, pos in pos_tags:
            if word.lower() in self.stop_words or not word.isalnum():
                continue
            if not pos.startswith(("NN", "VB", "JJ", "RB")):
                continue

            pred, prob = self.classify_word(word)
            if pred is not None:
                analyzed_count += 1
                attractive_score_sum += prob

                word_info = {"word": word, "pos": pos, "probability": float(prob)}

                if pred == 1:
                    results["attractive_words"].append(word_info)
                    results["summary"]["attractive_count"] += 1
                else:
                    results["non_attractive_words"].append(word_info)
                    results["summary"]["non_attractive_count"] += 1

        results["summary"]["total_analyzed"] = analyzed_count
        if analyzed_count > 0:
            results["summary"]["attractiveness_score"] = (
                attractive_score_sum / analyzed_count
            )

        return results

    def save_analysis(self, analysis, folder="analyses"):
        """Save each analysis result as a timestamped JSON file and open it in VS Code."""
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(folder, f"analysis_{timestamp}.json")

        with open(path, "w") as f:
            json.dump(analysis, f, indent=4)

        print(f"💾 Saved analysis to {path}")

        # 🚀 Automatically open the file in VS Code
        try:
            os.system(f"code {path}")
            print("🧠 Opened analysis in VS Code")
        except Exception as e:
            print(f"⚠️ Could not open in VS Code: {e}")


def print_analysis(analysis):
    """Pretty-print results to console"""
    print("\n----------------------------------------")
    print("✨ Attractive Words:")
    for w in analysis["attractive_words"]:
        print(f"  • {w['word']} ({w['probability']:.2f})")

    print("\n😐 Non-Attractive Words:")
    for w in analysis["non_attractive_words"]:
        print(f"  • {w['word']} ({w['probability']:.2f})")

    s = analysis["summary"]
    print("----------------------------------------")
    print(f"⭐ Total Analyzed: {s['total_analyzed']}")
    print(f"💫 Attractiveness Score: {s['attractiveness_score']:.2f}")
    print("----------------------------------------\n")
