from fastapi import FastAPI
import pickle
import torch

from MovieRecommend.MovieRecommendModel import MFAdvanced, MODEL_CONFIG, round_to_0p5


app = FastAPI(debug=True)
MODEL_DIR = "model"
MODEL_PATH = fr"{MODEL_DIR}/movieRecommendModel.pth"
ALREADY_RATED_PATH = fr"{MODEL_DIR}/AlreadyRated.pkl"
MOVIE_MAPPING_PATH = fr"{MODEL_DIR}/MovieMapping.pkl"
HOST = "127.0.0.1"
PORT = 8000

with open(ALREADY_RATED_PATH, "rb") as arf:
    already_rated = pickle.load(arf)


with open(MOVIE_MAPPING_PATH, "rb") as mmf:
    movie_mapping = pickle.load(mmf)


max_user_id = max(already_rated.keys())
max_movie_id = max(movie_mapping.keys())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MFAdvanced(
    num_users=MODEL_CONFIG["n_users"],
    num_items=MODEL_CONFIG["n_items"],
    emb_dim=MODEL_CONFIG["dim_size"],
    sigmoid=MODEL_CONFIG["sigmoid"],
    bias=MODEL_CONFIG["bias"],
    init=MODEL_CONFIG["init"],
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

from MovieRecommend import routes
