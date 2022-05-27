import cv2
import base64
import numpy as np
import model as Model


def loadBase64Img(encoded):
    nparr = np.fromstring(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def detect_face(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(cv2.data.haarcascades)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades +"haarcascade_frontalface_default.xml"
    )
    faces = detector.detectMultiScale(img_gray)

    x, y, w, h = faces[0]
    face = img[y : y + h, x : x + w]
    return face, faces[0]


def resize(img, size):
    old_size = img.shape[:2]
    ratio = size[0] / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    new_img = cv2.resize(img, new_size)

    delta_w = size[0] - new_size[1]
    delta_h = size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    new_img = cv2.copyMakeBorder(
        new_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return (new_img / 255).astype(np.float32)


def preprocess(img, input_shape):
    face_img, region_img = detect_face(img)

    face_img = resize(face_img, input_shape)

    return face_img


def calculate_cosine(predictions):
    p1, p2 = predictions

    a = np.matmul(np.transpose(p1), p2)
    b = np.sum(np.multiply(p1, p1))
    c = np.sum(np.multiply(p2, p2))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def verify(images):
    model = Model.build_model()
    input_shape = model.layers[0].input_shape[0][1:]

    prep_images = np.array([preprocess(img, input_shape) for img in images])

    # predictions = model.predict(prep_images, verbose=0)
    predictions = [model.predict([prep_img]) for prep_img in prep_images]

    distance = calculate_cosine(predictions)

    if distance <= 0.40:
        verified = True
    else:
        verified = False

    return {"verified": verified}, 200 if verified else 404

