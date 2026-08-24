import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- PATHS ---
PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "Rhine_proj")
BASE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "extracted_images")
BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "blip_results")

def run_blip_captioning():
    print("--- Loading BLIP Model ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    # Get all PDF subfolders inside the extracted_images directory
    if not os.path.exists(BASE_IMAGE_FOLDER):
        print(f"Error: Base image folder not found at {BASE_IMAGE_FOLDER}")
        return
        
    pdf_folders = [f for f in os.listdir(BASE_IMAGE_FOLDER) if os.path.isdir(os.path.join(BASE_IMAGE_FOLDER, f))]
    print(f"Found {len(pdf_folders)} PDF folders to process.\n")

    for index, pdf_base in enumerate(pdf_folders):
        # ==========================================================
        # TEMPORARY: Stop after the 1st PDF for testing
        # DELETE THIS 2-LINE BLOCK LATER FOR FULL PIPELINE
        if index >= 1:
            print(f"Skipping remaining PDFs (stopped at {pdf_base}).")
            break
        # ==========================================================

        print(f"========== STARTING BLIP FOR: {pdf_base} ==========")
        
        IMAGE_FOLDER = os.path.join(BASE_IMAGE_FOLDER, pdf_base)
        OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, pdf_base)
        
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        json_output_path = os.path.join(OUTPUT_FOLDER, "image_captions.json")

        # Get images, explicitly ignoring the 'junk_imgs' folder
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(image_files) == 0:
            print(f"-> SKIPPING {pdf_base}: No clean images found.")
            continue

        print(f"Found {len(image_files)} images to caption.")
        results_dict = {}

        for img_name in image_files:
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            
            try:
                raw_image = Image.open(img_path).convert('RGB')
                inputs = processor(raw_image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=50)
                    
                caption = processor.decode(out[0], skip_special_tokens=True)
                results_dict[img_name] = caption
                
                print(f"[{img_name}] -> {caption}")
                
            except Exception as e:
                print(f"FAILED on {img_name}: {e}")
                results_dict[img_name] = "ERROR_GENERATING_CAPTION"

        # Save to JSON
        print("--- Saving BLIP Output ---")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
            
        print(f"JSON saved to: {json_output_path}")
        print(f"========== COMPLETED BLIP FOR: {pdf_base} ==========\n")

if __name__ == "__main__":
    run_blip_captioning()