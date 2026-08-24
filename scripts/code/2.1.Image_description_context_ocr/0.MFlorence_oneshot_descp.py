import os
import json
import torch
import re
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

# --- PATHS ---
PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "Rhine_proj")
BASE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "extracted_images")
BASE_JSON_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "json_results")
BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "florence2_results")

def run_florence_captioning():
    print("--- Loading Florence-2 Model (Native Community Port) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")

    processor = AutoProcessor.from_pretrained("florence-community/Florence-2-large", trust_remote_code=True)
    model = Florence2ForConditionalGeneration.from_pretrained(
        "florence-community/Florence-2-large", 
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)

    if not os.path.exists(BASE_IMAGE_FOLDER):
        print(f"Error: Base image folder not found at {BASE_IMAGE_FOLDER}")
        return
        
    pdf_folders = [f for f in os.listdir(BASE_IMAGE_FOLDER) if os.path.isdir(os.path.join(BASE_IMAGE_FOLDER, f))]

    for index, pdf_base in enumerate(pdf_folders):

        print(f"\n========== STARTING FLORENCE-2 FOR: {pdf_base} ==========")
        
        IMAGE_FOLDER = os.path.join(BASE_IMAGE_FOLDER, pdf_base)
        OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, pdf_base)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        json_output_path = os.path.join(OUTPUT_FOLDER, "intermediate_captions_en.json")

        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        results_dict = {}

        for img_name in image_files:
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            
            # THE FIX: Use the native, hardwired task token. No complex instructions.
            task_prompt = "<MORE_DETAILED_CAPTION>"

            try:
                raw_image = Image.open(img_path).convert('RGB')
                
                inputs = processor(text=task_prompt, images=raw_image, return_tensors="pt").to(device, torch.float16)
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=512, # Increased slightly to allow for full map descriptions
                        do_sample=False,
                        num_beams=3
                    )
                    
                # Decode and clean the output
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                results_dict[img_name] = generated_text
                print(f"[{img_name}] -> {generated_text}")
                
            except Exception as e:
                print(f"FAILED on {img_name}: {e}")

        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
            
        print(f"========== COMPLETED FLORENCE-2 FOR: {pdf_base} ==========\n")

if __name__ == "__main__":
    run_florence_captioning()