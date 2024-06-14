import argparse
import glob
import os
import pickle
import pdf2image
import torch
import cv2
from text_extraction import extract_text
from create_grid import return_word_grid, select_tokenizer, create_grid_dict, create_mmocr_grid
from VGT.object_detection.ditod import add_vit_config
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from VGT.object_detection.ditod.VGTTrainer import DefaultPredictor
from mmocr.apis import MMOCRInferencer
from non_max import non_maximum_suppression
from lxml import etree
import pytesseract
from tqdm import tqdm
from labels import labels


def pdf_to_images(filename, dpi, experiment):
    pdf_name = os.path.basename(filename).split('.')[0]
    dirname = os.path.join(experiment, pdf_name, 'pages')
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    images = pdf2image.convert_from_path(filename, fmt='png', dpi=dpi)
    for i, image in enumerate(images):
        fp = os.path.join(dirname, f'page_{i}.png')
        image.save(fp)


def image_to_grids(image_path, tokenizer, inferencer):
    result = inferencer(image_path, return_vis=False)
    tokenizer = select_tokenizer(tokenizer)
    grid = create_mmocr_grid(tokenizer, result)
    save_path = os.path.join(*image_path.split('/')[:-2], 'grids', os.path.basename(image_path).split('.')[0] + '.pkl')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'wb') as file:
        pickle.dump(grid, file)


def pdf_to_grids(filename, tokenizer, experiment):
    pdf_name = os.path.basename(filename).split('.')[0]
    dirname = os.path.join(experiment, pdf_name, 'grids')
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # print(f'gen {filename}')
    word_grid = return_word_grid(filename)
    tokenizer = select_tokenizer(tokenizer)

    for i in range(len(word_grid)):
        try:
            grid = create_grid_dict(tokenizer, word_grid[i])
            with open(os.path.join(dirname, f'page_{i}.pkl'), 'wb') as file:
                pickle.dump(grid, file)
        except IndexError:
            print('error in ' + word_grid[i])
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to run VGT on pdf')
    parser.add_argument('--root',
                        type=str,
                        default='pdfs/AAPG-77',
                        help='path to input directory')

    parser.add_argument('--dataset',
                        type=str,
                        default='doclaynet',
                        help='pretrain dataset name: doclaynet or publaynet')

    parser.add_argument('--tokenizer',
                        type=str,
                        default='google-bert/bert-base-uncased',
                        help='tokenizer')

    parser.add_argument('--cfg',
                        help='cfg file path',
                        type=str,
                        default='configs/cascade/doclaynet_VGT_cascade_PTM.yaml')

    parser.add_argument("--opts",
                        help="Modify cfg options using the command-line 'KEY VALUE' pairs",
                        default=[],
                        nargs=argparse.REMAINDER)

    parser.add_argument('--dpi',
                        help='pdf conversion resolution',
                        type=int,
                        default=200)

    parser.add_argument('--output',
                        '-o',
                        help='output folder name',
                        type=str,
                        default='result/AAPG-ALL')

    parser.add_argument('--grid',
                        help='tool used for creating grids: pdfplumber or mmocr',
                        type=str,
                        default='pdfplumber')

    parser.add_argument('--ocr',
                        help='text extraction tool to use: mupdf, tesseract, or auto',
                        type=str,
                        default='mupdf')

    args = parser.parse_args()

    pdfs = glob.glob(os.path.join(args.root, '*.pdf'))

    if os.path.isdir(args.root):
        inputs = list()

        # Step 0: pdf preprocessing
        print('pre-processing files...')
        for pdf_path in pdfs:
            pdf_to_images(pdf_path, args.dpi, args.output)
            if args.grid == 'pdfplumber':
                pdf_to_grids(pdf_path, args.tokenizer, args.output)

        if args.grid == 'mmocr':
            infer = MMOCRInferencer(det='dbnetpp', rec='svtr-small')
            for pdf_path in pdfs:
                pdf_name = os.path.basename(pdf_path).split('.')[0]
                for i, image in enumerate(tqdm(
                        glob.glob(os.path.join(args.output, pdf_name, 'pages', '*.png')))):
                    image_to_grids(image, args.tokenizer, infer)

        # Step 1: instantiate config
        cfg = get_cfg()
        add_vit_config(cfg)
        cfg.merge_from_file(args.cfg)

        # Step 2: add model weights URL to config
        cfg.merge_from_list(args.opts)

        # Step 3: set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.DEVICE = device

        # Step 4: define model
        predictor = DefaultPredictor(cfg)

        # Step 6: run inference

        inputs = []
        for pdf_i, pdf in enumerate(pdfs):
            pdf_name = os.path.basename(pdf).split('.')[0]
            images = glob.glob(os.path.join(args.output, pdf_name, 'pages', '*.*'))
            # sort by page number
            images = sorted(images, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            for i, image_path in enumerate(tqdm(images)):
                img = cv2.imread(image_path)
                grid = os.path.join(*image_path.split('/')[:-2], 'grids', os.path.basename(image_path).split('.')[0]+'.pkl')
                page = image_path.split('/')[-1].split('.')[0]

                # load or create xml for the current pdf
                xml_path = os.path.join(*image_path.split('/')[:-2], 'output.xml')
                if not os.path.exists(xml_path):
                    root = etree.Element('output')
                else:
                    root = etree.parse(xml_path).getroot()

                if os.path.exists(grid):
                    # run inference
                    output = predictor(img, grid)["instances"]

                    # save VGT output
                    output_path = image_path.split('/')[:-2]
                    file_name = os.path.basename(image_path).split('.')[0] + '.pkl'
                    output_path = os.path.join(*output_path, 'outputs', file_name)

                    if not os.path.exists(os.path.dirname(output_path)):
                        os.makedirs(os.path.dirname(output_path))
                    pickle.dump(output, open(output_path, 'wb'))

                    # prepare folders to cropped bounding boxes from each page
                    cropped_image_dir = os.path.join(*image_path.split('/')[:-2], 'cropped_image')
                    if not os.path.exists(cropped_image_dir):
                        os.makedirs(cropped_image_dir)

                    cropped_text_dir = os.path.join(*image_path.split('/')[:-2], 'cropped_text')
                    if not os.path.exists(cropped_text_dir):
                        os.makedirs(cropped_text_dir)

                    page_element = etree.Element("page", number=page.split('_')[-1])

                    # extract bounding boxes
                    output = non_maximum_suppression(output, 0.3)
                    for j in range(len(output)):
                        x1 = int(output[j].pred_boxes.tensor.squeeze()[0].item())
                        y1 = int(output[j].pred_boxes.tensor.squeeze()[1].item())
                        x2 = int(output[j].pred_boxes.tensor.squeeze()[2].item())
                        y2 = int(output[j].pred_boxes.tensor.squeeze()[3].item())
                        figure = img[y1:y2, x1:x2, :]

                        image_label = {'doclaynet': 6, 'publaynet': 4}
                        table_label = {'doclaynet': 8, 'publaynet': 3}

                        if output[j].pred_classes.item() == image_label[args.dataset]:
                            crop_path = os.path.join(cropped_image_dir, f'{page}_box{j}.png')
                            cv2.imwrite(crop_path, figure)
                            box_type = 'image'
                            sub_type = ''
                            element_text = crop_path

                        elif output[j].pred_classes.item() == table_label[args.dataset]:
                            box_type = 'table'
                            sub_type = ''
                            element_text = ''

                        else: # text
                            crop_path = os.path.join(cropped_text_dir, f'{page}_box{j}.png')
                            cv2.imwrite(crop_path, figure)
                            h, w = img.shape[:2]
                            bbox = [x1/w, y1/h, x2/w, y2/h]

                            if args.ocr == 'mupdf':
                                element_text = extract_text(pdf, int(page.split('_')[-1]), bbox)

                            elif args.ocr == 'tesseract':
                                element_text = pytesseract.image_to_string(crop_path)

                            # try extraction first if empty use tesseract
                            elif args.ocr == 'auto':
                                element_text = extract_text(pdf, int(page.split('_')[-1]), bbox)
                                if element_text == '':
                                    element_text = pytesseract.image_to_string(crop_path)

                            box_type = 'text'
                            sub_type = labels[args.dataset][output[j].pred_classes.item()]

                        # write to xml
                        item_element = etree.Element("item",
                                                     block=str(j),
                                                     type=box_type,
                                                     subtype=sub_type,
                                                     x0=str(x1),
                                                     y0=str(y1),
                                                     x1=str(x2),
                                                     y1=str(y2),)

                        item_element.text = element_text if element_text is not None else ''
                        page_element.append(item_element)

                    # save to xml
                    root.append(page_element)
                    tree = etree.ElementTree(root)
                    tree.write(xml_path, pretty_print=True, xml_declaration=True)

    else:
        print('Root directory does not exist: {}'.format(args.root))
