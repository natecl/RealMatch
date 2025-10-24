from flask import Flask, render_template, jsonify, request
import speech_recognition as sr
import threading
import time
import os
import json

app = Flask(__name__)
r = sr.Recognizer()
recognized_text = ""  # global variable to hold last recognized text

# Ensure the text file exists
try:
    with open("text.txt", "a", encoding="utf-8"):
        pass
except Exception:
    pass

# Ensure conversation.json exists
if not os.path.exists("conversation.json"):
    try:
        with open("conversation.json", "w", encoding="utf-8") as f:
            json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "conversation": []}, f, indent=2)
    except Exception:
        pass


def record_audio():
    global recognized_text
    while True:
        try:
            with sr.Microphone() as source:
                print("Listening...")
                r.adjust_for_ambient_noise(source, duration=0.2)
                audio = r.listen(source)

                print("Recognizing...")
                text = r.recognize_google(audio)
                print(f"Recognized: {text}")
                recognized_text = text  # store last recognized text

                # Also append the recognized text to text.txt with a timestamp
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    line = f"[{timestamp}] {text}"
                    with open("text.txt", "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except Exception:
                            # fsync may not be available on all platforms; ignore if it fails
                            pass
                except Exception as e:
                    print(f"❌ Failed to write to text.txt: {e}")
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print(f"API error: {e}")


# Endpoint to receive text from the web client and save it to text.txt
@app.route("/save_text", methods=["POST"])
def save_text():
    global recognized_text
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    # update in-memory last recognized text
    recognized_text = text

    # append to text.txt with timestamp
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{timestamp}] {text}"
        with open("text.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
    except Exception as e:
        return jsonify({"error": f"Failed to write to text.txt: {e}"}), 500

    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/text")
def get_text():
    return jsonify({"text": recognized_text})


@app.route('/conversation')
def get_conversation():
    """Return the conversation.json contents as JSON."""
    try:
        if os.path.exists("conversation.json"):
            with open("conversation.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure shape
                if not isinstance(data, dict) or 'conversation' not in data:
                    data = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "conversation": []}
                return jsonify(data)
        # If file missing, return empty conversation (200) so client can render gracefully
        return jsonify({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "conversation": []})
    except Exception as e:
        # On error, return empty conversation but log error to console
        print(f"Error reading conversation.json: {e}")
        return jsonify({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "conversation": []})


@app.route('/save_conversation', methods=['POST'])
def save_conversation():
    """Save the current conversation.json (or fallback to text.txt) into a unique TXT file in past_convos/.

    Returns JSON with status and filename on success.
    """
    # Ensure past_convos directory exists
    try:
        os.makedirs('past_convos', exist_ok=True)
    except Exception as e:
        return jsonify({"error": f"Could not create past_convos folder: {e}"}), 500

    # Try to read conversation.json
    conv = None
    try:
        if os.path.exists('conversation.json'):
            with open('conversation.json', 'r', encoding='utf-8') as f:
                conv = json.load(f)
    except Exception:
        conv = None

    # Fallback: if no conversation.json or it's empty, try text.txt
    if not conv or not conv.get('conversation'):
        try:
            if os.path.exists('text.txt'):
                with open('text.txt', 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                # Convert lines into conversation entries with no precise times
                conv = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    'conversation': []
                }
                for l in lines:
                    # Try to keep existing bracketed timestamps if present
                    if l.startswith('[') and ']' in l:
                        # split off the first ']'
                        try:
                            timepart, textpart = l.split(']', 1)
                            timepart = timepart.lstrip('[').strip()
                            textpart = textpart.strip()
                            conv['conversation'].append({'time': timepart, 'text': textpart})
                        except Exception:
                            conv['conversation'].append({'time': '', 'text': l})
                    else:
                        conv['conversation'].append({'time': '', 'text': l})
        except Exception:
            conv = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), 'conversation': []}

    # If still empty, return error
    if not conv or not conv.get('conversation'):
        return jsonify({"error": "No conversation data available to save."}), 400

    # Build a human-readable TXT content
    try:
        header = f"Conversation saved: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n"
        body_lines = []
        for entry in conv.get('conversation', []):
            t = entry.get('time') or ''
            txt = entry.get('text') or ''
            if t:
                body_lines.append(f"[{t}] {txt}")
            else:
                body_lines.append(txt)

        content = header + "\n".join(body_lines) + "\n"

        # Unique filename: past_convos/convo_YYYYMMDD_HHMMSS.txt
        fname = f"convo_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.txt"
        outpath = os.path.join('past_convos', fname)
        with open(outpath, 'w', encoding='utf-8') as out:
            out.write(content)
    except Exception as e:
        return jsonify({"error": f"Failed to write conversation file: {e}"}), 500

    return jsonify({"status": "ok", "file": outpath})


if __name__ == "__main__":
    threading.Thread(target=record_audio, daemon=True).start()
    app.run(debug=True)
