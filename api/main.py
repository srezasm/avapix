from flask import Flask, request, jsonify
from model import Model  # Assuming you have a Model class that handles image encoding and decoding

app = Flask(__name__)
model = Model()  # Initialize your Model class

@app.route('/encode', methods=['POST'])
def encode():
    try:
        data = request.json
        text = data.get('text')

        if not text:
            return jsonify({'error': 'Text not provided'}), 400

        image_file_name = model.create_image(text)
        image_url = f'/static/images/{image_file_name}'  # Update the path to your image files

        return jsonify({'image_url': image_url}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/decode', methods=['POST'])
def decode():
    try:
        image_file = request.files.get('image')

        if not image_file:
            return jsonify({'error': 'Image file not provided'}), 400

        decoded_text = model.decode_image(image_file)
        
        return jsonify({'decoded_text': decoded_text}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
