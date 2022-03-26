from pydantic import BaseModel

class Movie(BaseModel):
    movie_id: int
    movie_title: str
    release_date: str
    score: int
