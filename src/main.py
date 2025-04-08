import cv2
import numpy as np
import httpx
import time
import convert as convert
import base64


def decode_base64(b64_img):
    # Data URIからBase64部分を抽出
    base64_data = b64_img.split(",")[1] if "," in b64_img else b64_img
    img_bytes = base64.b64decode(base64_data)
    image_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def grid_mosaic(image, size):
    aspect = image.shape[0] / image.shape[1]
    small = cv2.resize(
        image, (size, int(size * aspect)), interpolation=cv2.INTER_NEAREST
    )
    return small


cap = cv2.VideoCapture(0)
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

        img = grid_mosaic(img, 256)

        response = convert.upload(client, img)
        image_id = response.get("image_id") if response else None
        d_response = convert.dog(client, image_id)
        # kmeans処理
        k_response = convert.kmeans(client, image_id, k=16)
        c_response = convert.convert(client, image_id, k_response["cluster"])

        convert.delete(client, image_id)
        img = decode_base64(c_response["image"])
        result = cv2.resize(img, frame.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Converted Image", result)

        # cv2.imshow("Camera", frame)

        # qキーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.waitKey()
cv2.destroyAllWindows()
