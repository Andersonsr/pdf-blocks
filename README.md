# PDF blocks #
PDF Image and text extraction and pairing to generate datasets for OCR task and vision language models.

## Installation ##

1. Download and install [VGT](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/VGT) to execute document layout analysis. 
    ```angular2html
    git clone https://github.com/AlibabaResearch/AdvancedLiterateMachinery.git
    mv AdvancedLiterateMachinery/DocumentUnderstanding/VGT/ VGT/
    rm -r AdvancedLiterateMachinery/
    ```

2. Download pre trained model.
    ```angular2html
    wget https://github.com/AlibabaResearch/AdvancedLiterateMachinery/releases/download/v1.3.0-VGT-release/doclaynet_VGT_model.pth
    mkdir -p weights/vgt
    mv doclaynet_VGT_model.pth weights/vgt
    ```



3. Download [LayoutLM](https://huggingface.co/microsoft/layoutlm-base-uncased) required to run VGT
    ```angular2html
    git clone https://huggingface.co/microsoft/layoutlm-base-uncased
    ```

<b>Optionally</b> to run on scanned PDFs download [mmocr](https://github.com/open-mmlab/mmocr) text detector to generate the grids required to execute VGT.            
```angular2html
git clone https://github.com/open-mmlab/mmocr.git
```

## Usage ##