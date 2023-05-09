from flask import Blueprint, jsonify, request
# from api import calculation
from common.libs.generate_avatar import ml_api_method

api = Blueprint("api", __name__)

# @api.get("/")
@api.route("/", methods=["GET"])
def index():
    return jsonify({'column': "value"})

# # @api.post("/generate")
@api.route("/generate", methods=["GET"])
def detection():
    ml_api_method()
    return "value"