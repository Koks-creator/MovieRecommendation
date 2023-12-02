from pydantic import BaseModel, conint
from typing import List

from MovieRecommend.MovieRecommendModel import MODEL_CONFIG
from MovieRecommend import max_movie_id


class ListOfDicts(BaseModel):
    items: List[dict]


class User(BaseModel):
    user_id: conint(ge=0, le=MODEL_CONFIG["n_users"])


class UserRecommendations(User):
    recommended_movies: List[dict]


class UserRatingPredictionReq(User):
    movie_id: conint(ge=0, le=max_movie_id)


class UserRatingPredictionResponse(User):
    movie_id: conint(ge=0, le=max_movie_id)
    movie_name: str
    rating_raw: float
    rating_rounded: float
