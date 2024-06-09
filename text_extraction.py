import glob
import os.path
from math import ceil, floor
from lxml import etree
import cv2
from paddleocr import PaddleOCR, draw_ocr
import pytesseract
import fitz


def extract_text(pdf_path, page_num, bbox):
    document = fitz.open(pdf_path)
    page = document[page_num]
    w = page.rect.width
    h = page.rect.height
    # print(f'fitz height and width: {h} and {w}')
    bbox = [bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h]

    words = page.get_text("words")
    words = [w for w in words if fitz.Rect(w[:4]).intersects(bbox)]
    result_string = ''
    for word in words:
        result_string += word[4] + ' '
    return result_string


if __name__ == '__main__':
    pdf = 'pdfs/scan/Adams Atlas of Sedimentary Rocks.pdf'
    xml_path = 'result/xml-test-scan-mmocr/Adams Atlas of Sedimentary Rocks/output.xml'
    page_num = 6
    block_num = 0
    root = etree.parse(xml_path).getroot()
    block = root.xpath(f"//page[@number='{page_num}']/item[@block='{block_num}']")[0]

    page_png = cv2.imread('result/xml-test-scan-mmocr/Adams Atlas of Sedimentary Rocks/pages/page_6.png')
    h, w = page_png.shape[:2]
    print(f'original height and width: {h} {w}')

    bbox = [int(block.get('x0')) / w,
            int(block.get('y0')) / h,
            int(block.get('x1')) / w,
            int(block.get('y1')) / h]

    text1 = extract_text(pdf, page_num, bbox)
    if text1 == '':
        print('No text found')
    print('')
    print(text1)

