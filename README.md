# Vietnamese OCR

Ongoing Project

This is a project about Optical Character Recognition (OCR) in Vietnamese texts by using PaddleOCR and VietOCR.

# Outline

1. Text Detection
2. Text Recognition

# How to use

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

For Jupyter Notebook, you can explore and experiment with the code in predict.ipynb.

# Text Dectection

For the Text Detection task, the [DB algorithm](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_det_db_en.md) was used in the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) framework to localize text in the input image. To enhance the accuracy of Text Recognition, images cropped by the DB algorithm were padded.

# Text Recognition

For Text Recognitionm part, I used [VietOCR](https://github.com/pbcquoc/vietocr)-a popular framework for Vietnamese OCR task, baseded on Transformer OCR architecture.

# References

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [VietOCR](https://github.com/pbcquoc/vietocr)
