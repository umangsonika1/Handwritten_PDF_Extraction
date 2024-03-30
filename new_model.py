import streamlit as st
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from PIL import Image
import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from PIL import Image
import pytesseract
from detectron2.utils.visualizer import Visualizer
import cv2
import numpy as np
import fitz

from comer.datamodule import vocab
from comer.lit_comer import LitCoMER
from torchvision.transforms import ToTensor
import torch
from PIL import Image

from IPython.display import display


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def pdf_to_images(pdf_path):
    pdf_document = fitz.open(pdf_path)
    images = []

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        image = page.get_pixmap()
        images.append(image)

    return images


cfg = get_cfg()
cfg.merge_from_file("config_auto_label.yaml")
cfg.MODEL.WEIGHTS = "model_0015999.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the threshold for detection

st.title('Detectron 2 Streamlit App')
uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png", "pdf"])

ckpt = r'C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\lightning_logs\version_0\checkpoints\epoch=151-step=57151-val_ExpRate=0.6365.ckpt'

model = LitCoMER.load_from_checkpoint(ckpt)
model = model.eval()
device = torch.device("cpu")
model = model.to(device)


rgb_image = cv2.imread(r'C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\example\true.jpg')
st.image(rgb_image, caption='Input Image', use_column_width=True)

gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
_, black_white_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

cv2.imwrite(r'C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\true_gray_image.jpg', 255-gray_image)

image = cv2.imread(r'C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\true_gray_image.jpg')

# original_shape = image.shape
# print("Original Image shape:", original_shape)

# resized_image = cv2.resize(image, (original_shape[1] // 4, original_shape[0] // 4))
resized_image = cv2.resize(image, (600, 150))

gray_resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

cv2.imwrite(r'C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\resized_gray_true_image.png', gray_resized_image)




img_path = r"C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\resized_gray_true_image.png"
img = Image.open(img_path)

# Convert image to tensor
img_tensor = ToTensor()(img)
mask = torch.zeros_like(img_tensor, dtype=torch.bool)

# Perform model inference
hyp = model.approximate_joint_search(img_tensor.unsqueeze(0), mask)[0]
pred_latex = vocab.indices2label(hyp.seq)

# Display LaTeX output
st.write(f"Predicted LaTeX: {pred_latex}")

st.latex(pred_latex)



# if uploaded_file is not None:
#     file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
#     st.write(file_details)
#
#     if uploaded_file.type == "application/pdf":
#         pdf_contents = uploaded_file.read()
#         doc = fitz.open(stream=pdf_contents, filetype="pdf")
#         num_pages = doc.page_count
#         if num_pages > 0:
#             for page_number in range(num_pages):
#                 st.write(f"Processing page {page_number + 1}/{num_pages}")
#                 page = doc.load_page(page_number)
#                 img_bytes = page.get_pixmap().tobytes()
#
#                 image = np.array(bytearray(img_bytes), dtype=np.uint8)
#                 im_np = cv2.imdecode(image, cv2.IMREAD_COLOR)
#                 predictor = DefaultPredictor(cfg)
#                 outputs = predictor(im_np)
#                 v = Visualizer(im_np[:, :, ::-1], scale=1.0)
#                 out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#                 im_with_boxes = out.get_image()
#
#                 st.image(im_np[:, :, ::-1], caption=f'Page {page_number + 1}: Uploaded Image', use_column_width=True)
#                 st.image(im_with_boxes, caption=f'Page {page_number + 1}: Annotated Image', use_column_width=True)
#
#                 pred_boxes = outputs["instances"].to("cpu").pred_boxes.tensor
#                 pred_classes = outputs["instances"].to("cpu").pred_classes
#
#                 class_to_visualize = 1
#                 text_from_boxes = []
#
#                 for box, class_id in zip(pred_boxes, pred_classes):
#                     class_id = class_id.item()
#                     if class_id == class_to_visualize:
#                         x1, y1, x2, y2 = box.tolist()
#                         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                         cropped_img = im_np[y1:y2, x1:x2]
#                         pil_image = Image.fromarray(cropped_img)
#                         text = pytesseract.image_to_string(pil_image)
#                         text_from_boxes.append(text)
#
#                 for idx, text in enumerate(text_from_boxes):
#                     st.markdown(f"### Text from box {idx + 1} - Page {page_number + 1}")
#                     st.text(text)
#                     st.markdown("---")
#
#         else:
#             st.write("No pages found in the PDF file.")
#
#     elif uploaded_file.type.startswith("image/"):
#         image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
#         im_np = cv2.imdecode(image, cv2.IMREAD_COLOR)
#
#         predictor = DefaultPredictor(cfg)
#         outputs = predictor(im_np)
#         v = Visualizer(im_np[:, :, ::-1], scale=1.0)
#         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         im_with_boxes = out.get_image()
#
#         st.image(im_np[:, :, ::-1], caption='Uploaded Image', use_column_width=True)
#         st.image(im_with_boxes, caption='Annotated Image', use_column_width=True)
#
#         pred_boxes = outputs["instances"].to("cpu").pred_boxes.tensor
#         pred_classes = outputs["instances"].to("cpu").pred_classes
#
#         class_to_visualize = 1
#         text_from_boxes = []
#
#         for box, class_id in zip(pred_boxes, pred_classes):
#             class_id = class_id.item()
#             if class_id == class_to_visualize:
#                 x1, y1, x2, y2 = box.tolist()
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 cropped_img = im_np[y1:y2, x1:x2]
#                 pil_image = Image.fromarray(cropped_img)
#                 text = pytesseract.image_to_string(pil_image)
#                 text_from_boxes.append(text)
#
#         num_columns = len(text_from_boxes)
#         columns = st.columns(num_columns)
#
#         for idx, text in enumerate(text_from_boxes):
#             st.markdown(f"### Text from box {idx + 1}")
#             st.text(text)
#             st.markdown("---")  # Horizontal line for separation

        # for idx, text in enumerate(text_from_boxes):
        #     st.write(f"Text from box {idx + 1}: {text}")

        # Display each element from the list in a separate column
        # for idx, text in enumerate(text_from_boxes):
        #     columns[idx].write(f"Text from box {idx + 1}: {text}")
        # if text_from_boxes:
        #     for idx, text in enumerate(text_from_boxes, start=1):
        #         st.text(f"Text from box {idx}: {text}")
