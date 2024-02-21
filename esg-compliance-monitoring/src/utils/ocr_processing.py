from typing import List
import pytesseract
from pdf2image import convert_from_path
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using OCR.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def extract_text_from_images(image_paths: List[str]) -> List[str]:
    """
    Extracts text from a list of image files using OCR.

    Args:
        image_paths (List[str]): A list of paths to image files.

    Returns:
        List[str]: A list of extracted texts from the images.
    """
    extracted_texts = []
    for image_path in image_paths:
        text = pytesseract.image_to_string(image_path)
        extracted_texts.append(text)
    return extracted_texts

def save_extracted_text(text: str, output_path: str) -> None:
    """
    Saves the extracted text to a specified output file.

    Args:
        text (str): The text to save.
        output_path (str): The path to the output file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

def process_pdf_and_save_text(pdf_path: str, output_path: str) -> None:
    """
    Processes a PDF file to extract text and save it to a file.

    Args:
        pdf_path (str): The path to the PDF file.
        output_path (str): The path to save the extracted text.
    """
    extracted_text = extract_text_from_pdf(pdf_path)
    save_extracted_text(extracted_text, output_path)