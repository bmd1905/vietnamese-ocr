# Vietnamese OCR

This project is about Optical Character Recognition (OCR) in Vietnamese texts. It uses PaddleOCR and VietOCR frameworks to achieve this. PaddleOCR is a popular OCR framework that provides a wide range of OCR models and tools. VietOCR is a popular framework for Vietnamese OCR task, based on Transformer OCR architecture.

>Note that: this model is just a compiling model, which means that I have simply gathered scripts from models in order to create a cohesive and comprehensive result. The end-to-end project will be started in the near future.

# Outline

1. Text Detection
2. Text Recognition

# Text Dectection
Text detection is the process of locating text in an image or video and recognizing the presence of characters. The [DB algorithm](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_det_db_en.md) is a popular algorithm used in the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) framework to localize text in the input image. It works by detecting the text regions in the image and then grouping them into text lines. This algorithm is known for its high accuracy and speed.

To enhance the accuracy of Text Recognition, images cropped by the DB algorithm were padded. This is because the padding helps to ensure that the text is not cut off during the recognition process.

# Text Recognition

Text Recognition is the process of recognizing the text in an image or video. For Text Recognition part, you used [VietOCR](https://github.com/pbcquoc/vietocr), which is a popular framework for Vietnamese OCR task. It is based on Transformer OCR architecture. The Transformer OCR architecture is a combination of the CNN and Transformer models. The CNN model is used to extract features from the input image, while the Transformer model is used to recognize the text in the image. This architecture is known for its high accuracy and speed.

# Usage

Firstly, clone this repository by executing:

```
git clone https://github.com/bmd1905/vietnamese-ocr
```

After cloning the repository, download the required dependencies by running:

```
pip install -r requirement.txt
```

For command-line usage, execute the following script for inference:

```
python predict.py
    --img path/to/image
    --output path/of/output_image
```

For Jupyter Notebook, you can explore and experiment with the code at [predict.ipynb](https://github.com/bmd1905/vietnamese-ocr/blob/master/predict.ipynb).

# References

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [VietOCR](https://github.com/pbcquoc/vietocr)
