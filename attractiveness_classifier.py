"""
Attractiveness Classifier
Analyzes speech input for attractive vs non-attractive language
"""
import time

import numpy as np
from gensim.models import KeyedVectors
import joblib
from sklearn.ensemble import RandomForestClassifier
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class AttractivenessClassifier:
    def __init__(self):
        # Load pre-trained word vectors
        print("Loading word embeddings...")
        self.word_vectors = KeyedVectors.load_word2vec_format(
            'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz',
            binary=True
        )
        
        # Initialize and train the classifier
        self._initialize_classifier()
        
        # Load stop words
        self.stop_words = set(stopwords.words('english'))
        
    def _initialize_classifier(self):
        """Initialize and train the classifier with predefined attractive/non-attractive words"""
        # Example training data (can be expanded)
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
        
        # Prepare training data
        X = []
        y = []
        
        # Add attractive words
        for word in attractive_words:
            if word in self.word_vectors:
                X.append(self.word_vectors[word])
                y.append(1)
                
        # Add non-attractive words
        for word in non_attractive_words:
            if word in self.word_vectors:
                X.append(self.word_vectors[word])
                y.append(0)
        
        # Train the classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X, y)
    
    def get_word_vector(self, word):
        """Get word vector if available, otherwise return None"""
        try:
            return self.word_vectors[word.lower()]
        except KeyError:
            return None
    
    def classify_word(self, word):
        """Classify a single word as attractive (1) or non-attractive (0)"""
        vector = self.get_word_vector(word)
        if vector is not None:
            pred = self.classifier.predict([vector])[0]
            prob = self.classifier.predict_proba([vector])[0][1]  # Probability of being attractive
            return pred, prob
        return None, None
    
    def analyze_text(self, text):
        """
        Analyze text and return attractive/non-attractive words with their probabilities
        """
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Part of speech tagging
        pos_tags = nltk.pos_tag(tokens)
        
        results = {
            'attractive_words': [],
            'non_attractive_words': [],
            'summary': {
                'total_analyzed': 0,
                'attractive_count': 0,
                'non_attractive_count': 0,
                'attractiveness_score': 0.0
            }
        }
        
        analyzed_count = 0
        attractive_score_sum = 0
        
        for word, pos in pos_tags:
            # Skip stop words and non-content words
            if word.lower() in self.stop_words or not word.isalnum():
                continue
                
            # Only analyze nouns, verbs, adjectives, and adverbs
            if not pos.startswith(('NN', 'VB', 'JJ', 'RB')):
                continue
                
            pred, prob = self.classify_word(word)
            if pred is not None:
                analyzed_count += 1
                attractive_score_sum += prob
                
                word_info = {
                    'word': word,
                    'pos': pos,
                    'probability': float(prob)
                }
                
                if pred == 1:
                    results['attractive_words'].append(word_info)
                    results['summary']['attractive_count'] += 1
                else:
                    results['non_attractive_words'].append(word_info)
                    results['summary']['non_attractive_count'] += 1
        
        # Calculate summary statistics
        results['summary']['total_analyzed'] = analyzed_count
        if analyzed_count > 0:
            results['summary']['attractiveness_score'] = attractive_score_sum / analyzed_count
            
        return results

# Save analysis results to file
def save_analysis(analysis, filename="attractiveness_analysis.txt"):
    with open(filename, "a") as f:
        f.write("\n=== Attractiveness Analysis ===\n")
        f.write(f"Total words analyzed: {analysis['summary']['total_analyzed']}\n")
        f.write(f"Attractive words found: {analysis['summary']['attractive_count']}\n")
        f.write(f"Non-attractive words found: {analysis['summary']['non_attractive_count']}\n")
        f.write(f"Overall attractiveness score: {analysis['summary']['attractiveness_score']:.2f}\n\n")
        
        f.write("Attractive Words:\n")
        for word_info in analysis['attractive_words']:
            f.write(f"- {word_info['word']} ({word_info['probability']:.2f})\n")
            
        f.write("\nNon-attractive Words:\n")
        for word_info in analysis['non_attractive_words']:
            f.write(f"- {word_info['word']} ({word_info['probability']:.2f})\n")
        
        f.write("\n" + "="*30 + "\n")