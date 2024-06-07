import glob
from PIL import Image
import img2pdf


def topdf(img_path):
    # img = Image.open(img_path)
    pdf_bytes = img2pdf.convert(img_path)

    pdf_path = img_path.split('.')[0] + '.pdf'
    file = open(pdf_path, 'wb')
    file.write(pdf_bytes)
    file.close()


if __name__ == '__main__':
    path = 'result/VGT-doclay/AAPG Memoir 77_Colour Guide to the Petrography of Carbonate Rocks_Schole & Schole_2003/cropped_texts/*.png'
    texts = glob.glob(path)
    with open(
            'result/VGT-doclay/AAPG Memoir 77_Colour Guide to the Petrography of Carbonate Rocks_Schole & Schole_2003/concat.pdf', 'wb') as file:
        crop_bytes = img2pdf.convert(texts)
        file.write(crop_bytes)
        file.close()

