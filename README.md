# Tormenta #
PDF Image and text extraction and pairing to generate datasets for OCR task and vision language models.

## Installation ##

1. Download [VGT](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/VGT). 
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

4. Install dependencies
   ```angular2html
   pip install -r requirements.txt
   curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
   sudo apt-get install git-lfs
   ```
5. Build Detectron2 from source
   ```angular2html
   python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

6. Build MMOCR from source
   ```angular2html
   pip install -U openmim
   mim install mmengine
   mim install mmcv
   git clone https://github.com/open-mmlab/mmocr.git
   cd mmocr
   mim install -e .
   ``` 
   If you got a message like the following, check if your $CUDA_HOME variable is pointing to the correct path.
   MMCV searches for nvcc at $CUDA_HOME/bin/nvcc.
   ```angular2html
   error: [Errno 2] No such file or directory: '/root/anaconda3/envs/tormenta/bin/nvcc'
   ``` 

## Usage ##

