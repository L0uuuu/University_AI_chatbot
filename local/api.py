from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from unsloth import FastLanguageModel  # For optimized inference
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import re

# Initialize FastAPI app
app = FastAPI(
    title="LoRA Model API",
    description="API for an unsloth-trained model with LoRA adapters",
    version="1.0.0"
)

# Add CORS middleware to allow requests from your Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input data model
class ModelInput(BaseModel):
    prompt: str
    max_new_tokens: int = 256  # Default for legal responses
    temperature: float = 0.7   # Default for factual accuracy
    top_p: float = 0.9         # Default for legal texts
    repetition_penalty: float = 1.15  # Prevent repetitive phrases

# Load model and tokenizer globally (runs once when API starts)
model_path = "C:/Users/louai/OneDrive/Bureau/chapitre A/models/base/models--unsloth--Llama-3.2-3B-Instruct/snapshots/889fdd323e8975eec1e1e0fc821a2c818e6ad494"  # Replace with your base model path
lora_adapter_path = "./lora-adapter2"  # Replace with your LoRA adapter path

try:
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load and apply LoRA adapters
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    
    # Enable optimized inference
    model = FastLanguageModel.for_inference(model)  # 2x speed boost
    model.eval()  # Set to evaluation mode
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    raise Exception(f"Failed to load model: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Inference endpoint
@app.post("/generate")
async def generate_text(input_data: ModelInput):
    try:
        # Format input with chat template
        messages = [{"role": "user", "content": input_data.prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            max_length=2048,  # Match training length
            truncation=True
        ).to(device)
        
        # Define generation parameters
        generation_config = {
            "max_new_tokens": input_data.max_new_tokens,
            "temperature": input_data.temperature,
            "top_p": input_data.top_p,
            "repetition_penalty": input_data.repetition_penalty,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": True  # Critical for performance
        }
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                **generation_config
            )
        
        # Decode and clean output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        assistant_response = ""
        match = re.search(r"assistant\n\n(.*?)(\n\n|$)", generated_text, re.DOTALL)
        if match:
            assistant_response = match.group(1).strip()
        else:
            # Fallback: return the entire decoded text if assistant marker not found
            assistant_response = generated_text.strip()
        
        return {
            "input": input_data.prompt,
            "generated_text": assistant_response
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during generation: {str(e)}")

# Run the API
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Allows connections from other devices
        port=8000
    )