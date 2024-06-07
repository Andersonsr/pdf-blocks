import glob
import os.path
from math import ceil, floor
from lxml import etree
import cv2
from paddleocr import PaddleOCR, draw_ocr
import pytesseract
import fitz


def extract_text(pdf, page_num, bbox):
    document = fitz.open(pdf)
    page = document[page_num]
    w = page.rect.width
    h = page.rect.height
    print(f'fitz height and width: {h} and {w}')
    bbox = [bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h]

    words = page.get_text("words")
    words = [w for w in words if fitz.Rect(w[:4]).intersects(bbox)]
    result_string = ''
    for word in words:
        result_string += word[4] + ' '
    return result_string, page.get_textbox(bbox)


if __name__ == '__main__':
    pdf = 'pdfs/AAPG.pdf'
    xml_path = 'result/xml-test-multfig/AAPG/output.xml'
    page_num = 19
    block_num = 5
    root = etree.parse(xml_path).getroot()
    block = root.xpath(f"//page[@number='{page_num}']/item[@block='{block_num}']")[0]

    page_png = cv2.imread('result/xml-test-multfig/AAPG/pages/page_19.png')
    h, w = page_png.shape[:2]
    print(f'original height and width: {h} {w}')

    bbox = [int(block.get('x0')) / w,
            int(block.get('y0')) / h,
            int(block.get('x1')) / w,
            int(block.get('y1')) / h]

    text1, text2 = extract_text(pdf, page_num, bbox)

    print('')
    print(text1)
    print()
    print(text2)

