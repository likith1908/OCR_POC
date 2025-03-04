from flask import Flask, request, render_template, jsonify
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import fitz  # PyMuPDF
import google.generativeai as genai
import json 
import re
import difflib
import logging
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up device and model paths
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./GOT-OCR-2.0"
output_dir = "./output_images"
output_txt = "./ocr_output.txt"

# Check if model exists, download if not
if not os.path.exists(model_path):
    logging.info(f"Model not found at {model_path}. Downloading GOT-OCR-2.0...")
    try:
        model_name = "stepfun-ai/GOT-OCR-2.0-hf"
        model = AutoModelForImageTextToText.from_pretrained(model_name)
        model.save_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_name)
        processor.save_pretrained(model_path)
        logging.info("Model downloaded and saved successfully!")
    except Exception as e:
        logging.error(f"Error downloading model: {e}")
        exit()
else:
    logging.info(f"Model found at {model_path}. Skipping download.")

# Load GOT-OCR model and processor
logging.info(f"Loading GOT-OCR model from {model_path} on device {device}")
try:
    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)
    logging.info("Model and processor loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit()

# Configure Gemini API
logging.info("Gemini API configured")

# Lexicons remain the same
lexicons = {
    "laboratory_name": ["Aurobindo Central Lab", "R&D Lab 1", "Stability Testing Lab", "Microbiology Lab", "Quality Control Lab", "Analytical Testing Lab", "API Testing Lab", "Formulation Lab", "Bioanalytical Lab", "Regulatory Compliance Lab"],
    "department": ["Quality Control", "Research & Development", "Stability Studies", "API Analysis", "Finished Product Testing", "Microbiology", "Formulation Development", "Bioanalysis", "Regulatory Affairs", "Environmental Monitoring"],
    "analyst_name": ["Dr. A. Sharma", "Dr. P. Verma", "Dr. R. Iyer", "Dr. K. Patel", "Dr. M. Rao", "Dr. T. Das", "Dr. J. Kumar", "Dr. N. Reddy", "Dr. V. Singh", "Dr. S. Choudhury"],
    "reviewed_by": ["Dr. D. Nair", "Dr. P. Ghosh", "Dr. S. Jain", "Dr. R. Menon", "Dr. T. Mukherjee", "Dr. K. Ranganathan", "Dr. H. Gupta", "Dr. C. Srinivas", "Dr. M. Joshi", "Dr. S. Das"],
    "approved_by": ["Dr. B. Sen", "Dr. G. Mehta", "Dr. Y. Sharma", "Dr. P. Banerjee", "Dr. V. Nair", "Dr. R. Chatterjee", "Dr. S. Rajan", "Dr. K. Thakur", "Dr. J. Iqbal", "Dr. A. Thomas"],
    "sample_type": ["Raw Material", "Finished Product", "API", "Biological Sample", "Excipient"],
    "storage_conditions": ["Ambient", "Refrigerated", "Frozen"],
    "test_name": ["pH Analysis", "Assay by HPLC", "GC-MS Analysis", "Microbial Limit Test", "Dissolution", "Moisture Content", "Heavy Metals Analysis", "Particle Size Distribution", "FTIR Spectroscopy", "Stability Testing"],
    "test_method_reference": ["USP 40", "EP 10", "BP 2023", "IP 2024", "AOAC Official Methods", "ASTM D1193", "ICH Q2", "ISO 17025", "WHO Guidelines", "In-house SOP-001"],
}

FIELDS_TO_INCLUDE = [
    "general_information.report_id",
    "general_information.laboratory_name",
    "general_information.department",
    "general_information.date_of_report",
    "general_information.analyst_name",
    "general_information.reviewed_by",
    "general_information.approved_by",
    "sample_information.sample_id",
    "sample_information.batch_number",
    "sample_information.manufacturing_date",
    "sample_information.expiry_date",
    "test_details.test_name",
    "test_details.test_method_reference",
    "test_details.equipment_used",
    "test_details.test_start_date_time",
    "test_details.test_end_date_time",
    "test_details.reference_standard_used",
    "test_details.parameters[0].parameter",
    "test_details.parameters[0].observed_value",
    "test_details.parameters[0].unit",
    "test_details.parameters[0].specification_limit",
    "test_details.parameters[1].parameter",
    "test_details.parameters[1].observed_value",
    "test_details.parameters[1].unit",
    "test_details.parameters[1].specification_limit",
    "test_details.parameters[2].parameter",
    "test_details.parameters[2].observed_value",
    "test_details.parameters[2].unit",
    "test_details.parameters[2].specification_limit",
    "test_details.parameters[3].parameter",
    "test_details.parameters[3].observed_value",
    "test_details.parameters[3].unit",
    "test_details.parameters[3].specification_limit",
    "test_details.parameters[4].parameter",
    "test_details.parameters[4].observed_value",
    "test_details.parameters[4].unit",
    "test_details.parameters[4].specification_limit",
    "test_details.parameters[5].parameter",
    "test_details.parameters[5].observed_value",
    "test_details.parameters[5].unit",
    "test_details.parameters[5].specification_limit",
    "test_observations_results",
    "observations_analyst_comments[0]",
    "approval_signatures.analyst.name",
    "approval_signatures.analyst.date",
    "approval_signatures.reviewed_by.name",
    "approval_signatures.reviewed_by.date",
    "approval_signatures.approved_by.name",
    "approval_signatures.approved_by.date"
]

# Processing functions (removed calculate_metrics)
def pdf_to_images_pymupdf(pdf_path, output_dir, dpi=300, fmt='jpg'):
    logging.info(f"Converting PDF at {pdf_path} to images")
    pdf_doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    for page_num in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_num)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        output_path = os.path.join(output_dir, f"page_{page_num+1}.{fmt}")
        pix.save(output_path)
        image_paths.append(output_path)
        logging.info(f"Saved: {output_path}")
    pdf_doc.close()
    logging.info(f"PDF conversion complete, generated {len(image_paths)} images")
    return image_paths

def run_ocr_on_images(image_paths, output_file):
    logging.info("Starting OCR processing")
    combined_text = ""
    for image_path in image_paths:
        logging.info(f"Processing OCR on: {image_path}")
        try:
            inputs = processor(image_path, return_tensors="pt", format=True, crop_to_patches=True, max_patches=3).to(device)
            generate_ids = model.generate(
                **inputs,
                do_sample=False,
                tokenizer=processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=4096,
            )
            ocr_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            combined_text += f"\n=== Page: {image_path} ===\n{ocr_output}\n"
            logging.info(f"OCR completed for {image_path}")
        except RuntimeError as e:
            logging.warning(f"OCR failed: {e}. Falling back to CPU.")
            inputs = processor(image_path, return_tensors="pt", format=True, crop_to_patches=False).to("cpu")
            model.to("cpu")
            generate_ids = model.generate(
                **inputs,
                do_sample=False,
                tokenizer=processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=4096,
            )
            ocr_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            combined_text += f"\n=== Page: {image_path} (CPU Fallback) ===\n{ocr_output}\n"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(combined_text)
    logging.info(f"OCR result saved at: {output_file}")
    return combined_text

def convert_ocr_to_json(ocr_text):
    logging.info("Converting OCR text to JSON")
    prompt = f"""
    Convert the following OCR report into a structured JSON format.
    ### OCR Output:
    {ocr_text}
    ### JSON Format:
    {{
      "general_information": {{"report_id": "", "laboratory_name": "", "department": "", "date_of_report": "", "analyst_name": "", "reviewed_by": "", "approved_by": ""}},
      "sample_information": {{"sample_id": "", "sample_type": "", "batch_number": "", "manufacturing_date": "", "expiry_date": "", "storage_conditions": ""}},
      "test_details": {{
        "test_name": "",
        "test_method_reference": "",
        "parameters": [{{"parameter": "", "observed_value": null, "unit": "", "specification_limit": null, "result": ""}}],
        "equipment_used": "",
        "test_start_date_time": "",
        "test_end_date_time": "",
        "reference_standard_used": ""
      }},
      "test_observations_results": "",
      "observations_analyst_comments": [""],
      "approval_signatures": {{
        "analyst": {{"name": "", "date": ""}},
        "reviewed_by": {{"name": "", "date": ""}},
        "approved_by": {{"name": "", "date": ""}}
      }}
    }}
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    try:
        response = model.generate_content(prompt)
        json_result = response.text.strip()
        json_result_clean = re.sub(r"```json\n|\n```", "", json_result).strip()
        json_data = json.loads(json_result_clean)
        logging.info("JSON conversion successful")
        return json.dumps(json_data, indent=4)
    except Exception as e:
        logging.error(f"Error in JSON conversion: {e}")
        return "{}"

def find_best_match(value, lexicon):
    best_match = None
    best_ratio = 0
    for item in lexicon:
        ratio = difflib.SequenceMatcher(None, value.lower(), item.lower()).ratio()
        if ratio > best_ratio and ratio >= 0.45:
            best_ratio = ratio
            best_match = item
    return best_match

def normalize_date(value):
    if isinstance(value, str):
        date_pattern = r'(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{2,4})'
        match = re.match(date_pattern, value.strip())
        if match:
            day, month, year = match.groups()
            year = f"20{year}" if len(year) == 2 else year
            return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
    return value

def correct_with_lexicon(json_data):
    logging.info("Correcting JSON with lexicon")
    def process_value(value, key, parent_key=None):
        if isinstance(value, str):
            if key in lexicons:
                match = find_best_match(value, lexicons[key])
                if match:
                    logging.debug(f"Corrected {key}: '{value}' -> '{match}'")
                    return match
            elif key == "name" and parent_key in ["analyst", "reviewed_by", "approved_by"]:
                lexicon_key = "analyst_name" if parent_key == "analyst" else parent_key
                match = find_best_match(value, lexicons[lexicon_key])
                if match:
                    logging.debug(f"Corrected {parent_key}.name: '{value}' -> '{match}'")
                    return match
            if key in ["date_of_report", "manufacturing_date", "expiry_date", "date"]:
                normalized = normalize_date(value)
                if normalized != value:
                    logging.debug(f"Normalized {key}: '{value}' -> '{normalized}'")
                return normalized
        return value

    def traverse_json(data, parent_key=None):
        if isinstance(data, dict):
            for key, value in list(data.items()):
                data[key] = traverse_json(value, key)
                data[key] = process_value(data[key], key, parent_key)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                data[i] = traverse_json(item, parent_key)
        return data

    return traverse_json(json_data)

def process_pdf(pdf_file):
    if not pdf_file:
        return "Error: Please upload a PDF file"
    
    pdf_path = pdf_file
    image_paths = pdf_to_images_pymupdf(pdf_path, output_dir, dpi=300)
    ocr_text = run_ocr_on_images(image_paths, output_txt)
    json_result = convert_ocr_to_json(ocr_text)
    json_data = json.loads(json_result)
    corrected_json = correct_with_lexicon(json_data)
    corrected_json_str = json.dumps(corrected_json, indent=4)
    return corrected_json_str

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'Please upload a PDF file'}), 400
    
    pdf_file = request.files['pdf_file']
    
    if pdf_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save uploaded file temporarily
    pdf_path = os.path.join('uploads', pdf_file.filename)
    os.makedirs('uploads', exist_ok=True)
    pdf_file.save(pdf_path)
    
    # Process the file
    json_output = process_pdf(pdf_path)
    
    # Clean up temporary file
    os.remove(pdf_path)
    
    return jsonify({
        'json_output': json_output
    })

if __name__ == '__main__':
    logging.info("Starting Flask application")
    app.run(debug=True, host='0.0.0.0', port=5000)