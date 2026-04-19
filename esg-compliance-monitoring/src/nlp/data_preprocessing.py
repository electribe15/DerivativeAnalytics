from typing import List
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using OCR.
    
    Args:
        pdf_path (str): The path to the PDF file.
        
    Returns:
        str: Extracted text from the PDF.
    """
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def clean_extracted_text(text: str) -> str:
    """
    Cleans the extracted text by removing unnecessary characters and standardizing the format.
    
    Args:
        text (str): The raw extracted text.
        
    Returns:
        str: Cleaned text.
    """
    # Remove extra whitespace and newlines
    cleaned_text = ' '.join(text.split())
    return cleaned_text

def preprocess_regulatory_documents(pdf_folder: str) -> List[dict]:
    """
    Preprocesses all PDF regulatory documents in a specified folder.
    
    Args:
        pdf_folder (str): The folder containing PDF documents.
        
    Returns:
        List[dict]: A list of dictionaries containing the filename and cleaned text.
    """
    processed_data = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            extracted_text = extract_text_from_pdf(pdf_path)
            cleaned_text = clean_extracted_text(extracted_text)
            processed_data.append({'filename': filename, 'cleaned_text': cleaned_text})
    return processed_data