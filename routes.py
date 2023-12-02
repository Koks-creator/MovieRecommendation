from typing import Optional
from fastapi import HTTPException
import torch

from MovieRecommend import app, already_rated, movie_mapping, max_user_id, max_movie_id, model, \
    round_to_0p5, device, MODEL_CONFIG
from MovieRecommend.schemas import UserRecommendations, UserRatingPredictionReq, \
    UserRatingPredictionResponse, ListOfDicts


@app.get("/")
async def is_alive() -> dict:
    return {"msg": "Hello, I am alive"}


@app.get("/ratedMovies/<user_id>", response_model=ListOfDicts)
async def get_rated_movies(user_id: int) -> ListOfDicts:
    if user_id > max_user_id:
        raise HTTPException(422, f"Max user id is {max_user_id}")
    items = [{i: movie_mapping[i]} for i in already_rated[user_id]]
    return ListOfDicts(items=items)


@app.get("/getMovies", response_model=ListOfDicts)
async def get_movies() -> ListOfDicts:
    return ListOfDicts(items=[movie_mapping])


@app.get("/getRecommendations/<user_id>", response_model=UserRecommendations)
async def get_recommendations(user_id: int, limit: Optional[int] = 10) -> UserRecommendations:
    if user_id > max_user_id:
        raise HTTPException(422, f"Max user id is {max_user_id}")

    user_ratings = []
    already_rated_by_user = already_rated[user_id]
    all_movies = [i for i in range(MODEL_CONFIG["n_items"])]
    movies_to_check = list(set(all_movies) - set(already_rated_by_user))

    for item_id in range(len(movies_to_check)):
        predicted_rating = model(torch.tensor([user_id]).to(device), torch.tensor([item_id]).to(device))
        user_ratings.append((item_id, predicted_rating))

    user_ratings.sort(key=lambda x: x[1], reverse=True)

    recommended_movies = user_ratings[:limit]

    results = []
    for i, (item_id, predicted_rating) in enumerate(recommended_movies):
        results.append({
            i: {
                "MovieId": item_id,
                "MovieName": movie_mapping[item_id],
                "PredictedRating": round_to_0p5(predicted_rating.item()),
                "PredictedRatingRaw": predicted_rating.item(),
            }
        })

    return UserRecommendations(user_id=user_id, recommended_movies=results)


@app.post("/getRatingPrediction", response_model=UserRatingPredictionResponse)
def get_rating_prediction(data: UserRatingPredictionReq) -> UserRatingPredictionResponse:
    user_id = data.user_id
    movie_id = data.movie_id

    if user_id > max_user_id:
        raise HTTPException(422, f"Max user id is {max_user_id}")

    if movie_id > max_movie_id:
        raise HTTPException(422, f"Max movie id is {max_movie_id}")

    prediction_raw = model(torch.tensor([user_id]).to(device), torch.tensor([movie_id]).to(device)).item()
    prediction_round = round_to_0p5(prediction_raw)
    movie_name = movie_mapping[movie_id]

    return UserRatingPredictionResponse(
        user_id=user_id,
        movie_id=movie_id,
        movie_name=movie_name,
        rating_raw=prediction_raw,
        rating_rounded=prediction_round
    )

