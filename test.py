import pickle
import torch

from MovieRecommend.MovieRecommendModel import MFAdvanced, MODEL_CONFIG, round_to_0p5


MODEL_PATH = r"model/movieRecommendModel.pth"
MOVIE_MAPPING_PATH = r"model/MovieMapping.pkl"
ALREADY_RATED_PATH = r"model/AlreadyRated.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open(MOVIE_MAPPING_PATH, "rb") as mmf:
    movie_mapping = pickle.load(mmf)


with open(ALREADY_RATED_PATH, "rb") as arf:
    already_rated = pickle.load(arf)


movie_mapping_rev = {n: i for i, n in movie_mapping.items()}


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

movie_names = ["Highlander (1986)", "Road Warrior, The (Mad Max 2) (1981)", "American Pie (1999)", "Sixth Sense, The (1999)", "Ex Machina (2015)"]
users = [1, 1, 100, 100, 1]
actual_ratings = [3.5, 5.0, 4.0, 5.0, 3.5]
for movie, user, actual_rating in zip(movie_names, users, actual_ratings):
    movie_id = movie_mapping_rev[movie]
    predicted_rating = model(torch.tensor([user]).to(device), torch.tensor([movie_id]).to(device))
    rounded_pred = round_to_0p5([predicted_rating.item()])[0]
    print(f"User: {user}, Movie: {movie}, Predicted rating: {rounded_pred}, Actual rating: {actual_rating}")

user_id = 1
user_ratings = []
already_rated_by_user = already_rated[user_id]
all_movies = [i for i in range(MODEL_CONFIG["n_items"])]
movies_to_check = list(set(all_movies) - set(already_rated_by_user))

print(f"Recommendations for user: {1}")
for item_id in range(len(movies_to_check)):
    predicted_rating = model(torch.tensor([user_id]).to(device), torch.tensor([item_id]).to(device))
    user_ratings.append((item_id, predicted_rating))

user_ratings.sort(key=lambda x: x[1], reverse=True)

top_n = 10
recommended_movies = user_ratings[:top_n]

for i, (item_id, predicted_rating) in enumerate(recommended_movies):
    print(f"Recommendation {i + 1}: Movie {movie_mapping[item_id]} - Predicted Rating: {predicted_rating}")
