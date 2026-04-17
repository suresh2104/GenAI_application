import fitz
import cv2 # type: ignore
from PIL import Image
import tempfile
import os

def preprocess_file(uploaded_file):
    file_type = uploaded_file.type.split('/')[0]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
    try:
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        if file_type == "image":
            return [Image.open(temp_file.name)], None
        elif file_type == "application":  # PDF
            images, text_content = [], []
            with fitz.open(temp_file.name) as doc:
                for i, page in enumerate(doc):
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
                    text_content.append(f"--- Page {i + 1} ---\n{page.get_text()}")
            return images, "\n\n".join(text_content)
        elif file_type == "video":
            cap = cv2.VideoCapture(temp_file.name)
            frames, frame_count = [], 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % 30 == 0:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                frame_count += 1
            cap.release()
            return frames, None
        else:
            raise ValueError(f"Unsupported file type: {uploaded_file.type}")
    finally:
        os.unlink(temp_file.name)