"""
Real-time Speech Analysis for Attractiveness
Combines speech-to-text with attractiveness classification
"""

import speech_recognition as sr
from attractiveness_classifier import AttractivenessClassifier, print_analysis
from attractiveness_classifier import AttractivenessClassifier
import time
import sys

def setup_recognizer():
    """Initialize the speech recognizer"""
    recognizer = sr.Recognizer()
    return recognizer

def record_and_transcribe(recognizer):
    """Record audio and convert to text"""
    try:
        with sr.Microphone() as source:
            print("\nListening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio = recognizer.listen(source)

            print("Transcribing...")
            text = recognizer.recognize_google(audio)
            print(f"\nTranscribed Text: {text}")
            return text

    except sr.RequestError as e:
        print(f"Error with speech recognition service: {e}")
        return None
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    """Main function to run the speech analysis"""
    try:
        # Initialize speech recognition and classifier
        recognizer = setup_recognizer()
        classifier = AttractivenessClassifier()
        
        print("Speech Attractiveness Analyzer")
        print("Press Ctrl+C to exit")
        print("-" * 50)
        
        while True:
            # Record and transcribe speech
            text = record_and_transcribe(recognizer)
            
            if text:
                # Analyze the transcribed text
                analysis = classifier.analyze_text(text)
                
                # Print and save results
                print_analysis(analysis)
                classifier.save_analysis(analysis)
                
            time.sleep(0.1)  # Small delay to prevent CPU overuse
            
    except KeyboardInterrupt:
        print("\nExiting Speech Analysis...")
        sys.exit(0)

if __name__ == "__main__":
    main()