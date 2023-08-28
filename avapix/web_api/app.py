from flask import Flask, request, jsonify
import numpy as np

from avapix.web_api.wrapper import EmbedWrapper, DecodeWrapper

app = Flask(__name__)

embed_wrapper = EmbedWrapper()
decode_wrapper = DecodeWrapper()

@app.route("/embed", methods=["POST"])
def embed():
    try:
        data = request.json
        text = data.get("text")

        if not text:
            return jsonify({"error": "Text not provided"}), 400

        image_file_name = embed_wrapper.embed(text)
        
        return jsonify({"image_url": f"/static/{image_file_name}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/extract", methods=["POST"])
def extract():
    try:
        image_file = request.files.get("image").stream

        if not image_file:
            return jsonify({"error": "Image file not provided"}), 400

        decoded_text = decode_wrapper.extract(image_file)

        return jsonify({"decoded_text": decoded_text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5080, host='0.0.0.0')
