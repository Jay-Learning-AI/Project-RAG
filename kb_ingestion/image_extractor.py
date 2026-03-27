import os
import fitz
from docx import Document

TMP_DIR = "tmp_images"

def extract_images(file_path):
    os.makedirs(TMP_DIR, exist_ok=True)
    images = []

    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        for p in range(len(doc)):
            for i, img in enumerate(doc[p].get_images(full=True)):
                pix = fitz.Pixmap(doc, img[0])
                path = f"{TMP_DIR}/page{p}_{i}.png"
                pix.save(path)
                images.append(path)

    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                name = os.path.basename(rel.target_ref)
                path = f"{TMP_DIR}/{name}"
                with open(path, "wb") as f:
                    f.write(rel.target_part.blob)
                images.append(path)

    return images
