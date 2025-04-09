import cv2
import numpy as np
import httpx
import convert as convert
import base64
from PIL import Image
from PIL import ImageEnhance


def cv_to_base64(img):
    _, encoded = cv2.imencode(".png", img)
    img_str = base64.b64encode(encoded).decode("ascii")

    return img_str


def decode_base64(b64_img):
    # Data URIからBase64部分を抽出
    # base64_data = b64_img.split(",")[1] if "," in b64_img else b64_img
    img_bytes = base64.b64decode(b64_img)
    image_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def grid_mosaic(image, size):
    aspect = image.shape[0] / image.shape[1]
    small = cv2.resize(
        image, (size, int(size * aspect)), interpolation=cv2.INTER_NEAREST
    )
    return small


def saturation(image, value):
    img = Image.fromarray(image)
    enhancer = ImageEnhance.Color(img)
    result = enhancer.enhance(value)
    result = np.array(result)
    return result


cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# カメラが正常にオープンできたか確認
if not cap.isOpened():
    print("カメラがオープンできません")
    exit()

with httpx.Client() as client:
    # カメラの解像度を設定
    while True:
        ret, frame = cap.read()

        if not ret:
            print("カメラからの映像の取得に失敗しました")
            break
        img = cv2.GaussianBlur(
            frame,  # 入力画像
            (9, 9),  # カーネルの縦幅・横幅
            2,  # 横方向の標準偏差（0を指定すると、カーネルサイズから自動計算）
        )
        img = saturation(img, 2)

        response = convert.upload(client, img)
        image_id = response.get("image_id") if response else None
        convert.dog(client, image_id)
        img = decode_base64(convert.get(client, image_id)["image"])
        img = grid_mosaic(img, 256)

        convert._set(client, image_id, cv_to_base64(img))
        # kmeans処理
        k_response = convert.kmeans(client, image_id, k=16)
        convert.convert(client, image_id, k_response["cluster"])
        result = convert.get(client, image_id)

        convert.delete(client, image_id)
        img = decode_base64(result["image"])
        result = cv2.resize(img, frame.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Converted Image", result)

        # cv2.imshow("Camera", frame)

        # qキーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.waitKey()
cv2.destroyAllWindows()
