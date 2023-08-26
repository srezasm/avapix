from flask import Flask, request, jsonify
import torch
import numpy as np
from PIL import Image
import time

from avapix.common.models.avapix_model import AvapixModel
from avapix.common.constants import *
from avapix.common.strategy_context import StrategyContext
from avapix.common.versioned_strategies.strategy_v1 import StrategyV1
from avapix.web_api.image_utils import *
from avapix.web_api.api_constants import *

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AvapixModel().load_state_dict(torch.load(MODEL_PATH))

strategy = StrategyContext()
strategy.set_strategy(StrategyV1())


@app.route("/embed", methods=["POST"])
def embed():
    try:
        data = request.json
        text = data.get("text")

        if not text:
            return jsonify({"error": "Text not provided"}), 400

        img_arr = strategy.embed(text)
        img_tensor = numpy_to_tensor(img_arr, device)
        
        img_result = model(img_tensor)
        img_result = tensor_to_numpy(img_result)

        image_file_name = f"avatar_{str(int(time.time()))}.png"
        image_path = os.path.join(AVATAR_DIR, image_file_name)

        Image.fromarray(img_result.astype(np.uint8)).save(image_path)

        return jsonify({"image_url": f"/avatars/{image_file_name}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/decode", methods=["POST"])
def decode():
    try:
        image_file = request.files.get("image")

        if not image_file:
            return jsonify({"error": "Image file not provided"}), 400

        img_arr = np.array(Image.open(image_file).resize((8, 8)))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
