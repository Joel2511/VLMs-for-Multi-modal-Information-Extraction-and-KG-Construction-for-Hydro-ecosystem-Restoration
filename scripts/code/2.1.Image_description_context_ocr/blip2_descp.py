import os
import json
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# --- PATHS ---
PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "Rhine_proj")
BASE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "extracted_images")
BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "blip2_results")

def run_blip2_captioning():
    print("--- Loading BLIP-2 Model (FLAN-T5-XL) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")

    # 1. Load the instruction-tuned FLAN-T5 processor and model
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    
    # Load in float16 to safely fit inside the V100 GPU's 16GB memory limit
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", 
        torch_dtype=torch.float16
    ).to(device)

    if not os.path.exists(BASE_IMAGE_FOLDER):
        print(f"Error: Base image folder not found at {BASE_IMAGE_FOLDER}")
        return
        
    pdf_folders = [f for f in os.listdir(BASE_IMAGE_FOLDER) if os.path.isdir(os.path.join(BASE_IMAGE_FOLDER, f))]
    print(f"Found {len(pdf_folders)} PDF folders to process.\n")

    for index, pdf_base in enumerate(pdf_folders):
        # ==========================================================
        # TEMPORARY: Stop after the 1st PDF for testing
        # DELETE THIS BLOCK LATER FOR FULL PIPELINE
        if index >= 1:
            print(f"Skipping remaining PDFs (stopped at {pdf_base}).")
            break
        # ==========================================================

        print(f"========== STARTING BLIP-2 FOR: {pdf_base} ==========")
        
        IMAGE_FOLDER = os.path.join(BASE_IMAGE_FOLDER, pdf_base)
        OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, pdf_base)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        json_output_path = os.path.join(OUTPUT_FOLDER, "image_captions.json")

        image_files = [f for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(image_files) == 0:
            print(f"-> SKIPPING {pdf_base}: No clean images found.")
            continue

        print(f"Found {len(image_files)} images to caption.")
        results_dict = {}

        # 2. Define the strict instruction prompt
        raw_prompt = "Question: Describe this technical document image in detail, explicitly naming any maps, charts, or topographical features you see. Answer:"

        for img_name in image_files:
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            
            try:
                raw_image = Image.open(img_path).convert('RGB')
                
                inputs = processor(images=raw_image, text=raw_prompt, return_tensors="pt").to(device, torch.float16)
                
                with torch.no_grad():
                    # 3. Upgraded generation parameters to prevent hallucination and looping
                    out = model.generate(
                        **inputs, 
                        max_new_tokens=150,
                        repetition_penalty=1.5  # Forces the model to use new words instead of looping
                    )
                    
                caption = processor.decode(out[0], skip_special_tokens=True).strip()
                
                # 4. The Slicer: Cleanly remove the prompt text if the model spits it back out
                clean_prompt = raw_prompt.lower()
                if caption.lower().startswith(clean_prompt):
                    caption = caption[len(raw_prompt):].strip()
                elif "Answer:" in caption:
                    caption = caption.split("Answer:")[-1].strip()
                    
                results_dict[img_name] = caption
                
                print(f"[{img_name}] -> {caption}")
                
            except Exception as e:
                print(f"FAILED on {img_name}: {e}")
                results_dict[img_name] = "ERROR_GENERATING_CAPTION"

        print("--- Saving BLIP-2 Output ---")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
            
        print(f"JSON saved to: {json_output_path}")
        print(f"========== COMPLETED BLIP-2 FOR: {pdf_base} ==========\n")

if __name__ == "__main__":
    run_blip2_captioning()