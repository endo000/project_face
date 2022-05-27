from flask import Flask, jsonify, request, make_response
import vision
app = Flask(__name__)


@app.route("/verify", methods=["POST"])
def verify():
    req = request.get_json()
    resp_obj = jsonify({"verified": False}, 200)

    try:
        images = (vision.loadBase64Img(req[img]) for img in ("img1", "img2"))
        resp_obj = vision.verify(images)
    except Exception as err:
        resp_obj = ({"verified": False, "error": str(err)}, 404)

    return resp_obj


@app.route("/detectface", methods=["POST"])
def detectface():
    req = request.get_json()
    resp_obj = ({"detected": False}, 404)
    try:
        img = vision.loadBase64Img(req["img"])
        faces = (vision.detect_face(img), 200)
        if(len(faces) > 0):
            resp_obj = ({"detected": True}, 200)

    except Exception as err:
        resp_obj = ({"detected": False, "error": str(err)}, 404)

    return resp_obj


if __name__ == "__main__":
    app.run(host="0.0.0.0")

