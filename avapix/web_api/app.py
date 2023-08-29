from flask import Flask, request, jsonify
import os

from avapix.web_api.api_helper import Helper

app = Flask(__name__)

helper = Helper()


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

        return jsonify({"decoded_text": decoded_text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5080, host="0.0.0.0")
