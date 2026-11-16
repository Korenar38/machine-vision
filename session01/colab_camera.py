"""Веб-камера в Google Colab.

Colab работает на удалённом сервере, поэтому cv2.VideoCapture не видит вашу камеру.
Кадры снимает браузер (JavaScript) и передаёт их в Python по одному.

    for frame, pressed in frames(buttons=["Сохранить"]):  # цикл идёт до кнопки «Стоп»
        show(frame)

    frame = snapshot()  # один кадр
"""

import base64
import json

import cv2
import numpy as np
from google.colab.output import eval_js
from IPython.display import Javascript, display

JS = """
window.mvCamera = {
  async start(buttons) {
    const box = document.createElement('div');
    const video = document.createElement('video');
    const screen = document.createElement('img');
    const panel = document.createElement('div');
    video.muted = true;
    video.playsInline = true;
    video.style.cssText = 'position:absolute;width:1px;height:1px;opacity:0';
    screen.style.cssText = 'display:block;max-width:100%;min-height:40px';
    screen.onload = () => {
      if (window.google && google.colab && google.colab.output) {
        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
      }
    };
    panel.style.margin = '6px 0';
    this.pressed = null;
    this.stopped = false;
    for (const name of [...buttons, 'Стоп']) {
      const button = document.createElement('button');
      button.textContent = name;
      button.style.margin = '0 4px 4px 0';
      button.onclick = () => {
        if (name === 'Стоп') this.stopped = true;
        else this.pressed = name;
      };
      panel.appendChild(button);
    }
    box.append(video, screen, panel);
    document.body.appendChild(box);
    this.stream = await navigator.mediaDevices.getUserMedia({video: true});
    video.srcObject = this.stream;
    await video.play();
    if (!video.videoWidth) {
      await new Promise((done) => { video.onloadeddata = done; });
    }
    await new Promise((done) => setTimeout(done, 700));  // камера настраивает экспозицию
    this.box = box;
    this.video = video;
    this.screen = screen;
    this.canvas = document.createElement('canvas');
    this.canvas.width = video.videoWidth;
    this.canvas.height = video.videoHeight;
  },
  grab(quality) {
    if (this.stopped) return null;
    const pressed = this.pressed;
    this.pressed = null;
    this.canvas.getContext('2d').drawImage(this.video, 0, 0);
    return {image: this.canvas.toDataURL('image/jpeg', quality), pressed};
  },
  show(image) {
    this.screen.src = image;
  },
  stop() {
    if (this.stream) this.stream.getTracks().forEach((track) => track.stop());
    if (this.box) this.box.querySelector('video').remove();
    this.stream = null;
  },
};
"""


def _decode(data_url):
    """Картинка из браузера (data URL) → массив BGR, как у cv2.imread."""
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def show(picture):
    """Показать кадр на экране камеры, запущенной через frames()."""
    ok, buf = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 80])
    image = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    eval_js(f"window.mvCamera.show({json.dumps(image)})")


def frames(buttons=(), quality=0.7):
    """Выдавать кадры, пока не нажата кнопка «Стоп».

    На каждом шаге даёт пару: кадр (BGR) и название нажатой кнопки (или None).
    Браузер спросит разрешение на камеру: нажмите «Разрешить».
    """
    display(Javascript(JS))
    eval_js(f"window.mvCamera.start({json.dumps(list(buttons))})")
    try:
        while True:
            answer = eval_js(f"window.mvCamera.grab({quality})")
            if answer is None:
                break
            yield _decode(answer["image"]), answer["pressed"]
    finally:
        eval_js("window.mvCamera.stop()")


def snapshot():
    """Снять один кадр и выключить камеру."""
    stream = frames()
    frame, _ = next(stream)
    stream.close()
    return frame
