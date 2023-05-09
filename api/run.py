import os

from flask import Flask

from api import api
from api.config import config

config_name = os.environ.get("CONFIG", "local")

app = Flask(__name__)

app.config.from_object(config[config_name])
app.register_blueprint(api)

