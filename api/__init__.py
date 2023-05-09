from flask import Blueprint, jsonify, request
# from api import calculation
from common.libs.generate_avatar import ml_api_method

api = Blueprint("api", __name__)

@api.get("/")
def index():
    return jsonify({'column': "value"})


@api.post("/generate")
def detection():
    ml_api_method()
    return "value"