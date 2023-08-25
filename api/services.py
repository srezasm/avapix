import torch
from model import AvaPixModel
from settings import *
import utils
import random
import os
from PIL import Image
import time
import numpy as np

class Services():
    def __init__(self):
        if not os.path.isdir(AVATAR_DIR):
            os.mkdir(AVATAR_DIR)

        self.device = ('cuda'
                       if torch.cuda.is_available()
                       else 'cpu')

        self.model = AvaPixModel()
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.to(self.device)

    def embed_text(self, text):
        random_seed = random.randint(0, 255)

        input_img = (
            utils
            .embed_raw_img_v1(text, random_seed)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        avatar_arr = self.model(input_img)
        avatar_arr = utils.img_tensor_to_numpy(avatar_arr)
        avatar_arr = np.kron(avatar_arr, np.ones((30, 30, 1)))

        avatar_file_name = os.path.join(AVATAR_DIR, f'avatar_{str(int(time.time()))}.png')

        Image.fromarray(avatar_arr.astype(np.uint8)).save(avatar_file_name)

        return avatar_file_name
    
    def decode_image(self, image):
        img = Image.open(image)
        img = img.resize((8, 8))

        img = np.array(img)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        text = utils.extract_text(img)

        return text
