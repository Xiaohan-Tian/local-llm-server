from flask import Blueprint, jsonify

route_hi = Blueprint('route_hi', __name__)

@route_hi.route('/hi')
def hi():
    return jsonify({"hi": "server is running."})
