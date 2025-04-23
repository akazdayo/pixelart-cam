import cv2
import numpy as np
import httpx
from typing import Optional


def upload(client: httpx.Client, image: np.ndarray) -> Optional[dict]:
    """
    OpenCVで読み込んだ画像をPOSTリクエストで送信する

    Args:
        client: HTTPXクライアント
        image: OpenCVで読み込んだ画像データ（numpy.ndarray）

    Returns:
        レスポンスのJSONデータ、エラー時はNone
    """
    try:
        # numpy配列をPNGフォーマットのバイトデータに変換
        _, img_encoded = cv2.imencode(".png", image)
        img_bytes = img_encoded.tobytes()

        # ファイル名とMIMEタイプを指定してPOSTリクエスト
        files = {"upload_image": ("image.png", img_bytes, "image/png")}

        response = client.post(
            "http://localhost:8000/v1/images/upload", files=files, timeout=30.0
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as e:
        print(f"HTTPエラーが発生しました: {e}")
        return None
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None


def kmeans(client, image_id, k=8):
    response = client.post(
        "http://localhost:8000/v1/images/convert/kmeans?image_id="
        + image_id
        + "&k="
        + str(k),
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def convert(client: httpx.Client, image_id, palette):
    response = client.post(
        "http://localhost:8000/v1/images/convert?image_id=" + image_id,
        data=palette,
        headers={"Content-Type": "application/json"},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def delete(client: httpx.Client, image_id):
    client.get(
        "http://localhost:8000/v1/images/delete/" + image_id,
        timeout=30.0,
    )


def dog(client: httpx.Client, image_id):
    response = client.post(
        "http://localhost:8000/v1/images/convert/dog?image_id=" + image_id,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def morphology(client: httpx.Client, image_id):
    response = client.post(
        "http://localhost:8000/v1/images/convert/morphology?image_id=" + image_id,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def _set(client: httpx.Client, image_id, value):
    response = client.post(
        f"http://localhost:8000/v1/images/set?image_id={image_id}",
        json={"image_data": value},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def get(client: httpx.Client, image_id):
    response = client.get(
        "http://localhost:8000/v1/images/" + image_id,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()
