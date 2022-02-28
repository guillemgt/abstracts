'''
    This files processes the json files output by 'evaluate.py' (or rather, the first such file) and produces PNG images of the titles and abstracts. It requires the user to have pdflatex and dvipng installed.
'''

import os
import shutil
from pdf2image import convert_from_path
from PIL import Image
import json

TEXT_RESULTS_PATH = "results/"
IMAGES_PATH = "images/"

if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)



def add_padding(img, vertical_padding, horizontal_padding):
    width, height = img.size
    new_width = width + horizontal_padding
    new_height = height + vertical_padding
    result = Image.new(img.mode, (new_width, new_height), (255, 255, 255))
    result.paste(img, (horizontal_padding // 2, vertical_padding // 2))
    return result

def create_image_from_latex(name, latex):
    f = open("tmp/a.tex","w+")
    f.write("\\documentclass{article}\n\\usepackage{chemfig}\n\\begin{document}\n")
    f.write(latex+"\n")
    f.write(r"\end{document}")
    f.close()

    latex_error_code = os.system("pdflatex -output-directory=tmp/ --output-format=dvi -halt-on-error tmp/a.tex")
    if latex_error_code:
        #print("Latex did not compile :(")
        return False
    os.system("dvipng -T tight -D 2048 -o tmp/a.png tmp/a.dvi")
    image = Image.open("tmp/a.png")
    image = add_padding(image, 2*1280 - image.height % 4, 2*1280 - image.width % 4)
    image = image.convert("RGB")
    image = image.resize((image.width // 4, image.height // 4), resample=Image.BILINEAR)
    image.save(os.path.join(IMAGES_PATH, name + ".png"))
    return True

if not os.path.exists("tmp"):
    os.makedirs("tmp")

with open(os.path.join(TEXT_RESULTS_PATH, "results_0.json"), "r") as f:
    i = 0
    data = json.load(f)

    for sample in data:
        if "arxiv" in sample["title"] or "arxiv" in sample["abstract"]:
            continue
        if create_image_from_latex(str(i), "\\pagenumbering{gobble}\\date{}\n\\author{Todd R. Matthews}\n\\title{" + sample["title"] + "}\n\\maketitle\\abstract{" + sample["abstract"] + "}"):
            i += 1

shutil.rmtree("tmp")