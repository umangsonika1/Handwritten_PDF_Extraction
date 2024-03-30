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
import base64
import numpy as np
import fitz
import cv2
from comer.datamodule import vocab
from comer.lit_comer import LitCoMER
from torchvision.transforms import ToTensor
import torch
from PIL import Image
from IPython.display import display
from PIL import ImageOps
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from streamlit.components.v1 import html

ckpt = r'C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\lightning_logs\version_0\checkpoints\epoch=151-step=57151-val_ExpRate=0.6365.ckpt'

model = LitCoMER.load_from_checkpoint(ckpt)
model = model.eval()
device = torch.device("cpu")
model = model.to(device)


def latex_result(cropped_img):
    gray_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, black_white_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    #
    cv2.imwrite(r'C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\true_gray_image.jpg',
                255 - gray_image)
    image = cv2.imread(r'C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\true_gray_image.jpg')
    resized_image = cv2.resize(image, (600, 150))
    gray_resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(r'C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\resized_gray_true_image.png',
                gray_resized_image)
    img_path = r"C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\resized_gray_true_image.png"
    img = Image.open(img_path)
    img_tensor = ToTensor()(img)
    mask = torch.zeros_like(img_tensor, dtype=torch.bool)
    hyp = model.approximate_joint_search(img_tensor.unsqueeze(0), mask)[0]
    pred_latex = vocab.indices2label(hyp.seq)
    # st.write(f"{pred_latex}")
    # st.latex(pred_latex)
    return pred_latex

def generate_html(sorted_line_boxes, output_file, img, im_np):
    html_content = "<html>\n<head>\n<script type='text/javascript' async src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML'></script>\n<script type='text/x-mathjax-config'>MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']], displayMath: [['$$','$$'], ['\\[','\\]']], processEscapes: true, ignoreClass: 'text'}});</script>\n</head>\n<body>\n"

    for line_idx, (box) in enumerate(sorted_line_boxes):
            line_html = "  <div>\n"
            # st.write(box)
            class_id, x1, y1, x2, y2 = box

            if 0 <= x1 < x2 < img.shape[1] and 0 <= y1 < y2 < img.shape[0]:
                t = 0

                if class_id == 1:
                    cropped_image = img[y1:y2, x1:x2]
                    pred_latex_f = latex_result(cropped_image)
                    # res = st.latex(pred_latex_f)
                    latex_expression = f"$\\displaystyle {pred_latex_f}$"
                    line_html += f"    <span style='text-align: center; color: black; font-size:12px'>{latex_expression}</span>\n"

                elif class_id == 0:
                    cropped_image = img[y1:y2-3, x1:x2]
                    pil_image = Image.fromarray(cropped_image)
                    # Save image and get its path
                    image_name = f"image_{line_idx}_{t}.png"
                    t+=1
                    dir_path = r"C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\output\paper"
                    image_path = os.path.join(dir_path, image_name)
                    pil_image.save(image_path)
                    line_html += f"    <img src='{image_path}'/>\n"

                else:
                    cropped_image = im_np[y1:y2, x1:x2]
                    pil_line = Image.fromarray(cropped_image)
                    # Add whitespace padding to the PIL Image
                    padding = 10  # You can adjust the padding value
                    padded_image = ImageOps.expand(pil_line, border=padding, fill='white')
                    # Pass the padded image to pytesseract
                    text = pytesseract.image_to_string(pil_line, config='--psm 6')
                    extra_symbols = ['$', '%', '#', '@', '!', '>', '<', '/', 'Ч', 'о', 'С']

                    # extra_symbols = ['$', '%', '#', '@', '!', '>', '<', '/']
                    for symbol in extra_symbols:
                        text = text.replace(symbol, '')


                    line_html += f"    <span class='text'>{text}</span>\n"

            line_html += "  </div>\n"
            html_content += line_html

    html_content += "</body>\n</html>"

    # Save the HTML content to the output file
    with open(output_file, "w") as f:
        f.write(html_content)


def process_image(uploaded_file):
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    im_np = cv2.imdecode(image, cv2.IMREAD_COLOR)
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im_np)
    pred_boxes = outputs["instances"].to("cpu").pred_boxes.tensor
    pred_classes = outputs["instances"].to("cpu").pred_classes

    elements = []
    latexlst = []
    bounding_box = []
    img_cat = []

    for box, class_id in zip(pred_boxes, pred_classes):
        class_id = class_id.item()
        if class_id in [4]:
            x1, y1, x2, y2 = box.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped_img = im_np[y1:y2, x1:x2]

            # pil_image = Image.fromarray(cropped_img)
            text = pytesseract.image_to_string(cropped_img)
            elements.append((cropped_img, text, (x1, y1, x2, y2)))
            bounding_box.append((class_id, x1, y1, x2, y2))

        if class_id in [0]:
            x1, y1, x2, y2 = box.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped_img = im_np[y1:y2, x1:x2]
            img_cat.append((cropped_img, (x1, y1, x2, y2)))
            bounding_box.append((class_id, x1, y1, x2, y2))

        if class_id in [1]:
            x1, y1, x2, y2 = box.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped_img = im_np[y1:y2, x1:x2]

            # pil_image = Image.fromarray(cropped_img)
            text = pytesseract.image_to_string(cropped_img)
            latexlst.append((cropped_img, text, (x1, y1, x2, y2)))
            bounding_box.append((class_id, x1, y1, x2, y2))

    elements.sort(key=lambda elem: (elem[2][1], elem[2][0]))
    latexlst.sort(key=lambda elem: (elem[2][1], elem[2][0]))
    img_cat.sort(key=lambda elem: (elem[1][1], elem[1][0]))
    bounding_box.sort(key=lambda elem: (elem[2], elem[1]))

    return im_np, elements, latexlst, bounding_box, img_cat

def pdf_to_images(pdf_path):
    pdf_document = fitz.open(pdf_path)
    images = []
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        image = page.get_pixmap()
        images.append(image)

    return images

def process_pdf(uploaded_file):
    image = np.array(bytearray(uploaded_file), dtype=np.uint8)
    im_np = cv2.imdecode(image, cv2.IMREAD_COLOR)
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im_np)
    pred_boxes = outputs["instances"].to("cpu").pred_boxes.tensor
    pred_classes = outputs["instances"].to("cpu").pred_classes

    elements = []
    latexlst = []
    bounding_box = []
    img_cat = []
    for box, class_id in zip(pred_boxes, pred_classes):
        class_id = class_id.item()
        if class_id in [4]:
            x1, y1, x2, y2 = box.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped_img = im_np[y1:y2, x1:x2]
            # pil_image = Image.fromarray(cropped_img)
            text = pytesseract.image_to_string(cropped_img)
            elements.append((cropped_img, text, (x1, y1, x2, y2)))
            bounding_box.append((class_id, x1, y1, x2, y2))

        if class_id in [0]:
            x1, y1, x2, y2 = box.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped_img = im_np[y1:y2, x1:x2]
            img_cat.append((cropped_img, (x1, y1, x2, y2)))
            bounding_box.append((class_id, x1, y1, x2, y2))

        if class_id in [1]:
            x1, y1, x2, y2 = box.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped_img = im_np[y1:y2, x1:x2]
            # pil_image = Image.fromarray(cropped_img)
            text = pytesseract.image_to_string(cropped_img)
            latexlst.append((cropped_img, text, (x1, y1, x2, y2)))
            bounding_box.append((class_id, x1, y1, x2, y2))

    elements.sort(key=lambda elem: (elem[2][1], elem[2][0]))
    latexlst.sort(key=lambda elem: (elem[2][1], elem[2][0]))
    img_cat.sort(key=lambda elem: (elem[1][1], elem[1][0]))
    bounding_box.sort(key=lambda elem: (elem[2], elem[1]))

    return im_np, elements, latexlst, bounding_box, img_cat

cfg = get_cfg()
cfg.merge_from_file("config_auto_label.yaml")
cfg.MODEL.WEIGHTS = "model_0015999.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the threshold for detection

st.title('Detectron 2 Streamlit App')
uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png", "pdf"])

def generate_download_link(uploaded_file):
    html_file_path = r"C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\output\newf2.html"

    with open(html_file_path, "rb") as file:
        file_contents = file.read()

    file_name = "newf2.html"
    b64 = base64.b64encode(file_contents).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{file_name}">Download File</a>'
    return href

def generate_download_link_pdf(uploaded_file):
    html_file_path = r"C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\output\newpdf2.html"

    with open(html_file_path, "rb") as file:
        file_contents = file.read()

    file_name = "newpdf2.html"
    b64 = base64.b64encode(file_contents).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{file_name}">Download File</a>'
    return href

def get_sharable_link(file_path):
    with open(file_path, "rb") as file:
        file_contents = file.read()

    file_b64 = base64.b64encode(file_contents).decode()
    href = f'<a href="data:text/html;base64,{file_b64}" download="newf2.html">Download HTML File</a>'
    return href



if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    st.write(file_details)

    if uploaded_file.type == "application/pdf":
        pdf_contents = uploaded_file.read()
        doc = fitz.open(stream=pdf_contents, filetype="pdf")
        num_pages = doc.page_count
        if num_pages > 0:
            for page_number in range(num_pages):
                st.write(f"Processing page {page_number + 1}/{num_pages}")
                page = doc.load_page(page_number)

                img_bytes = page.get_pixmap().tobytes()
                # st.image(img_bytes)

                im_np_f, elements, latexlst, bounding_box, img_cat = process_pdf(img_bytes)

                bbox_lst = []
                for idx, (class_id, x1, y1, x2, y2) in enumerate(bounding_box, start=1):
                    bbox_lst.append((class_id, x1, y1, x2, y2))

                for line_idx, (box) in enumerate(bbox_lst):
                    # st.write(box)
                    class_id, x1, y1, x2, y2 = box

                    if 0 <= x1 < x2 < im_np_f.shape[1] and 0 <= y1 < y2 < im_np_f.shape[0]:
                        t = 0

                        if class_id == 1:
                            cropped_image = im_np_f[y1:y2, x1:x2]
                            pred_latex_f = latex_result(cropped_image)
                            # res = st.latex(pred_latex_f)
                            latex_expression = f"$\\displaystyle {pred_latex_f}$"
                            # st.write(f"{pred_latex_f}")
                            st.latex(pred_latex_f)

                        elif class_id == 0:
                            cropped_image = im_np_f[y1:y2 - 3, x1:x2]
                            pil_image = Image.fromarray(cropped_image)
                            st.image(cropped_image[:, :, ::-1], caption=f'Element {idx}', use_column_width=True)
                            # Save image and get its path
                            # image_name = f"image_{line_idx}_{t}.png"
                            # t += 1
                            # dir_path = r"C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\output\paper"
                            # image_path = os.path.join(dir_path, image_name)
                            # pil_image.save(image_path)

                        else:
                            cropped_image = im_np_f[y1:y2, x1:x2]
                            pil_line = Image.fromarray(cropped_image)
                            # Add whitespace padding to the PIL Image
                            padding = 10  # You can adjust the padding value
                            padded_image = ImageOps.expand(pil_line, border=padding, fill='white')
                            # Pass the padded image to pytesseract
                            text = pytesseract.image_to_string(pil_line, config='--psm 6')
                            extra_symbols = ['$', '%', '#', '@', '!', '>', '<', '/', 'Ч', 'о', 'С']

                            # extra_symbols = ['$', '%', '#', '@', '!', '>', '<', '/']
                            for symbol in extra_symbols:
                                text = text.replace(symbol, '')

                            st.write(f"{text}")





                # for idx, (cropped_img, text, _) in enumerate(latexlst, start=1):
                #     st.image(cropped_img[:, :, ::-1], caption=f'Element {idx}', use_column_width=True)
                #     gray_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                #     _, black_white_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
                #     cv2.imwrite(r'C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\true_gray_image.jpg',
                #                 255 - gray_image)
                #
                #     image = cv2.imread(
                #         r'C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\true_gray_image.jpg')
                #     resized_image = cv2.resize(image, (600, 150))
                #     gray_resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                #     cv2.imwrite(
                #         r'C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\resized_gray_true_image.png',
                #         gray_resized_image)
                #     img_path = r"C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\resized_gray_true_image.png"
                #     img = Image.open(img_path)
                #     img_tensor = ToTensor()(img)
                #     mask = torch.zeros_like(img_tensor, dtype=torch.bool)
                #     hyp = model.approximate_joint_search(img_tensor.unsqueeze(0), mask)[0]
                #     pred_latex = vocab.indices2label(hyp.seq)
                #     st.write(f"Predicted LaTeX: {pred_latex}")
                #     st.latex(pred_latex)
                #
                # for idx, (cropped_img, text, _) in enumerate(elements, start=1):
                #     st.image(cropped_img[:, :, ::-1], caption=f'Element {idx}', use_column_width=True)
                #     extra_symbols = ['$', '%', '#', '@', '!', '>', '<', '/']
                #     for symbol in extra_symbols:
                #         text = text.replace(symbol, '')
                #     # return text
                #
                #     st.write(f"{text}")
                #
                # for idx, (cropped_img, _) in enumerate(img_cat, start=1):
                #     st.image(cropped_img[:, :, ::-1], caption=f'Element {idx}', use_column_width=True)

                # bbox_lst = []
                # for idx, (class_id, x1, y1, x2, y2) in enumerate(bounding_box, start=1):
                #     bbox_lst.append((class_id, x1, y1, x2, y2))
                generate_html(bbox_lst, r"C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\output\newpdf2.html", im_np_f, im_np_f)

            st.title("HTML File Downloader")
            html_file_path = r"C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\output\newpdf2.html"

            with open(html_file_path, "rb") as file:
                file_contents = file.read()

            file_name = "newpdf2.html"
            download_button = st.download_button(label="Download HTML File", data=file_contents,
                                                 file_name=file_name, mime="text/html", key='file')

            download_link = generate_download_link_pdf(uploaded_file)
            st.markdown(download_link, unsafe_allow_html=True)

            # st.markdown(get_sharable_link(html_file_path), unsafe_allow_html=True)
            st.write(
                "Check out this [link](file:///C:/Users/Brigosha_Guest/PycharmProjects/pythonProject1/output/newpdf2.html)")

        else:
            st.write("No pages found in the PDF file.")

    elif uploaded_file.type.startswith("image/"):
        im_np_f, elements, latexlst, bounding_box, img_cat = process_image(uploaded_file)

        bbox_lst = []
        for idx, (class_id, x1, y1, x2, y2) in enumerate(bounding_box, start=1):
            bbox_lst.append((class_id, x1, y1, x2, y2))


        for line_idx, (box) in enumerate(bbox_lst):
            # st.write(box)
            class_id, x1, y1, x2, y2 = box

            if 0 <= x1 < x2 < im_np_f.shape[1] and 0 <= y1 < y2 < im_np_f.shape[0]:
                t = 0

                if class_id == 1:
                    cropped_image = im_np_f[y1:y2, x1:x2]
                    pred_latex_f = latex_result(cropped_image)
                    # res = st.latex(pred_latex_f)
                    latex_expression = f"$\\displaystyle {pred_latex_f}$"
                    # st.write(f"{pred_latex_f}")
                    st.latex(pred_latex_f)

                elif class_id == 0:
                    cropped_image = im_np_f[y1:y2 - 3, x1:x2]
                    pil_image = Image.fromarray(cropped_image)
                    st.image(cropped_image[:, :, ::-1], caption=f'Element {idx}', use_column_width=True)
                    # Save image and get its path
                    # image_name = f"image_{line_idx}_{t}.png"
                    # t += 1
                    # dir_path = r"C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\output\paper"
                    # image_path = os.path.join(dir_path, image_name)
                    # pil_image.save(image_path)

                else:
                    cropped_image = im_np_f[y1:y2, x1:x2]
                    pil_line = Image.fromarray(cropped_image)
                    # Add whitespace padding to the PIL Image
                    padding = 10  # You can adjust the padding value
                    padded_image = ImageOps.expand(pil_line, border=padding, fill='white')
                    # Pass the padded image to pytesseract
                    text = pytesseract.image_to_string(pil_line, config='--psm 6')
                    extra_symbols = ['$', '%', '#', '@', '!', '>', '<', '/', 'Ч', 'о', 'С']

                    # extra_symbols = ['$', '%', '#', '@', '!', '>', '<', '/']
                    for symbol in extra_symbols:
                        text = text.replace(symbol, '')

                    st.write(f"{text}")

        # for idx, (cropped_img, text, _) in enumerate(latexlst, start=1):
        #     st.image(cropped_img[:, :, ::-1], caption=f'Element {idx}', use_column_width=True)
        #
        #     gray_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        #     _, black_white_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        #     #
        #     cv2.imwrite(r'C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\true_gray_image.jpg',
        #                 255 - gray_image)
        #     image = cv2.imread(r'C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\true_gray_image.jpg')
        #     resized_image = cv2.resize(image, (600, 150))
        #     gray_resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        #     cv2.imwrite(r'C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\resized_gray_true_image.png',
        #                 gray_resized_image)
        #     img_path = r"C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\ans\resized_gray_true_image.png"
        #     img = Image.open(img_path)
        #     img_tensor = ToTensor()(img)
        #     mask = torch.zeros_like(img_tensor, dtype=torch.bool)
        #     hyp = model.approximate_joint_search(img_tensor.unsqueeze(0), mask)[0]
        #     pred_latex = vocab.indices2label(hyp.seq)
        #     st.write(f"{pred_latex}")
        #     st.latex(pred_latex)
        #
        # for idx, (cropped_img, text, _) in enumerate(elements, start=1):
        #     st.image(cropped_img[:, :, ::-1], caption=f'Element {idx}', use_column_width=True)
        #     extra_symbols = ['$', '%', '#', '@', '!', '>', '<', '/']
        #     for symbol in extra_symbols:
        #         text = text.replace(symbol, '')
        #     # return text
        #
        #     st.write(f"{text}")
        #
        # for idx, (cropped_img, _) in enumerate(img_cat, start=1):
        #     st.image(cropped_img[:, :, ::-1], caption=f'Element {idx}', use_column_width=True)
        #
        # bbox_lst = []
        # for idx, (class_id, x1, y1, x2, y2) in enumerate(bounding_box, start=1):
        #     bbox_lst.append((class_id, x1, y1, x2, y2))




        generate_html(bbox_lst, r"C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\output\newf2.html" ,im_np_f, im_np_f)


        st.title("HTML File Downloader")
        html_file_path = r"C:\Users\Brigosha_Guest\PycharmProjects\pythonProject1\output\newf2.html"

        with open(html_file_path, "rb") as file:
            file_contents = file.read()

        file_name = "newf2.html"
        download_button = st.download_button(label="Download HTML File", data=file_contents, file_name=file_name, mime="text/html", key='file')

        download_link = generate_download_link(uploaded_file)
        st.markdown(download_link, unsafe_allow_html=True)

        # st.markdown(get_sharable_link(html_file_path), unsafe_allow_html=True)
        st.write("Check out this [link](file:///C:/Users/Brigosha_Guest/PycharmProjects/pythonProject1/output/newf2.html)")
