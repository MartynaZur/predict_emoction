from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import joblib
import traceback
from scipy.io.wavfile import write

app = Flask(__name__)
CORS(app)  # Dodaj CORS

# Wczytanie modelu (upewnij się, że ścieżka do modelu jest poprawna)
model = joblib.load("D:\\noc_naukowców\\emotion_recognition_model.joblib")


@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    try:
        # Odbieranie pliku audio
        if 'audio_data' not in request.files:
            return jsonify({"error": "No audio file part"}), 400
        
        audio_file = request.files['audio_data']
        if audio_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        file_path = "user_recording.wav"
        audio_file.save(file_path)
        
        # Testowanie pliku
        print(f"File saved to {file_path}")

        # Przewidywanie emocji
        y, sr = librosa.load(file_path, sr=48000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs = np.mean(mfccs.T, axis=0)
        mfccs = mfccs.reshape(1, -1)

        prediction = model.predict(mfccs)

        # Mapowanie numerów na emocje
        label_map = {
            1: 'Neutral',
            2: 'Calm',
            3: 'Happiness',
            4: 'Sadness',
            5: 'Anger',
            6: 'Fear',
            7: 'Disgust',
            8: 'Surprise'
        }
        emotion = label_map.get(prediction[0], 'Unknown')

        return jsonify({"emotion": emotion})

    except Exception as e:
        print("Error:", str(e))
        traceback.print_exc()  # To print detailed traceback in case of an exception
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
