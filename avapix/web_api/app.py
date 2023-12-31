from flask import Flask, request, jsonify
import os

import avapix.web_api.api_helper as helper
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/embed", methods=["POST"])
def embed():
    try:
        data = request.json
        text = data.get("text")

        if not text:
            return jsonify({"error": "Text not provided"}), 400

        # TODO: optionally get random seed from user
        # TODO: optionally get version from user
        image_file_name = helper.embed(text)

        return jsonify({"image_url": f"/static/{image_file_name}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/extract", methods=["POST"])
def extract():
    try:
        image_file = request.files.get("image").stream

        if not image_file:
            return jsonify({"error": "Image file not provided"}), 400

        decoded_text = helper.extract(image_file)

        return jsonify({"text": decoded_text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/is-ready", methods=["GET", "POST"])
def is_ready():
    try:
        text = "is ready?"

        file_name = helper.embed(text)
        assert file_name

        file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "static/" + file_name)
        )

        decoded_text = helper.extract(file_path)
        assert decoded_text == text

        os.remove(file_path)

        return jsonify(success=True), 200
    except:
        return jsonify(success=False), 500


if __name__ == "__main__":
    app.run(debug=False, port=80, host="0.0.0.0")
