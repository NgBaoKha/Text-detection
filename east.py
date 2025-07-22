import cv2
import numpy as np
import threading
import requests
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from paddleocr import PaddleOCR

# ==== EAST TEXT DETECTION ====
def decode_predictions(scores, geometry, conf_threshold=0.5):
    (numRows, numCols) = scores.shape[2:4]
    rects, confidences = [], []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        x0, x1, x2, x3, angles = [geometry[0, i, y] for i in range(5)]

        for x in range(numCols):
            if scoresData[x] < conf_threshold:
                continue

            offsetX, offsetY = x * 4.0, y * 4.0
            angle = angles[x]
            cos, sin = np.cos(angle), np.sin(angle)
            h = x0[x] + x2[x]
            w = x1[x] + x3[x]

            endX = int(offsetX + cos * x1[x] + sin * x2[x])
            endY = int(offsetY - sin * x1[x] + cos * x2[x])
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))

    return rects, confidences

def detect_text(image, net):
    H, W = image.shape[:2]
    newW, newH = (320, 320)
    rW, rH = W / newW, H / newH
    resized = cv2.resize(image, (newW, newH))

    blob = cv2.dnn.blobFromImage(resized, 1.0, (newW, newH),
                                 (123.68, 116.78, 103.94), True, False)
    net.setInput(blob)
    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid",
                                    "feature_fusion/concat_3"])
    rects, confidences = decode_predictions(scores, geometry)
    indices = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)

    results = []
    if indices is not None and len(indices) > 0:
        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            startX, startY, endX, endY = rects[i]
            startX, startY = int(startX * rW), int(startY * rH)
            endX, endY = int(endX * rW), int(endY * rH)
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            roi = image[startY:endY, startX:endX]
            results.append(roi)


    return image, results

# ==== UI & LOGIC ====
class TextDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EAST Text Detector")
        self.panel = Label(root)
        self.panel.pack()

        # Buttons & entries
        btn_frame = Frame(root)
        btn_frame.pack()

        Button(btn_frame, text="Chọn ảnh", command=self.load_image).grid(row=0, column=0)
        Label(btn_frame, text="Camera URL:").grid(row=0, column=1)
        self.url_entry = Entry(btn_frame, width=30)
        self.url_entry.insert(0, "http://192.168.100.156:8080/")
        self.url_entry.grid(row=0, column=2)
        Button(btn_frame, text="Bắt đầu", command=self.start_camera).grid(row=0, column=3)
        Button(btn_frame, text="Tắt", command=self.stop_camera).grid(row=0, column=4)
        Button(btn_frame, text="Lưu Text", command=self.save_text).grid(row=0, column=5)
        Button(btn_frame, text="Clear", command=self.clear_text).grid(row=0, column=6)

        # Text box
        Label(root, text="Text đã phát hiện (có thể sửa):").pack()
        self.text_box = Text(root, height=6, width=80)
        self.text_box.pack(padx=10, pady=5)

        self.cap = None
        self.running = False
        self.net = cv2.dnn.readNet("frozen_east_text_detection.pb")
        self.last_frame = None
        self.last_results = []
        self.ocr = PaddleOCR(lang='latin')

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        image = cv2.imread(path)
        self.last_frame, self.last_results = detect_text(image, self.net)
        self.show_image(self.last_frame)

    def show_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ImageTk.PhotoImage(Image.fromarray(image))
        self.panel.configure(image=image)
        self.panel.image = image

    def start_camera(self):
        url = self.url_entry.get()
        if not url.endswith("/video"):
            url = url.rstrip("/") + "/video"  # Đảm bảo đúng định dạng URL stream
        self.camera_url = url
        self.running = True
        threading.Thread(target=self.camera_loop, daemon=True).start()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def camera_loop(self):
        try:
            stream = requests.get(self.camera_url, stream=True)
            bytes_data = b""
            for chunk in stream.iter_content(chunk_size=1024):
                if not self.running:
                    break
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1 and b > a:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    img_array = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if frame is None:
                        continue

                    frame = cv2.resize(frame, (640, 480))
                    self.last_frame, self.last_results = detect_text(frame.copy(), self.net)
                    self.show_image(self.last_frame)
        except Exception as e:
            print(f"Lỗi đọc camera: {e}")

    def save_text(self):
        if not self.last_results:
            messagebox.showinfo("Thông báo", "Chưa có vùng chữ nào được phát hiện.")
            return

        text_output = ""

        for roi in self.last_results:
            # PaddleOCR cần ảnh là đường dẫn hoặc numpy array RGB
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            result = self.ocr.ocr(rgb, cls=True)

            for line in result[0]:
                _, (text, score) = line
                if score > 0.5:
                    text_output += f"{text.strip()}\n"

        self.text_box.delete("1.0", END)
        self.text_box.insert(END, text_output.strip())

    
    def clear_text(self):
        self.text_box.delete("1.0", END)

# ==== Run App ====
if __name__ == "__main__":
    root = Tk()
    app = TextDetectorApp(root)
    root.mainloop()
