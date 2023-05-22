from flask import Blueprint, jsonify, request
# from api import calculation
from common.libs.generate_avatar import ml_api_method

api = Blueprint("api", __name__)


FIXED_INFERENCE_ID='6430fd17-a7b2-4fd9-9e4a-407e789162d9'
FIXED_USER_ID = '7fe65695-a8e0-4b29-8ba8-9256d64904d3'
# @api.get("/")
@api.route("/", methods=["GET"])
def index():
    return jsonify({'column': "value"})

# # @api.post("/generate")
@api.route("/generate", methods=["GET"])
def detection():
    ml_api_method(FIXED_USER_ID, FIXED_INFERENCE_ID)
    return "value"