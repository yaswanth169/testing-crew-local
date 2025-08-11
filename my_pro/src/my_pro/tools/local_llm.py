import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Optional, Dict, Any, List
import os
from crewai.llms.base import BaseLLM
from crewai.llms.types import LLMResult
from ..config import get_llm_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalLLaMALLM(BaseLLM):
    """Crew AI LLM integration for local LLaMA model."""
    
    def __init__(self, model_path: str = None, **kwargs):
        """Initialize the local LLaMA LLM."""
        super().__init__(**kwargs)
        
        # Load configuration
        try:
            config = get_llm_config("local_llama")
            self.model_path = model_path or config.get("model_path", "C:/devhome/projects/models/meta-llama-Llama-3-2-1B-Instruct/meta-llama-Llama-3-2-1B-Instruct")
            self.max_tokens = config.get("max_tokens", 512)
            self.temperature = config.get("temperature", 0.7)
            self.top_p = config.get("top_p", 0.9)
            self.repetition_penalty = config.get("repetition_penalty", 1.1)
        except Exception as e:
            logger.warning(f"Could not load LLM configuration, using defaults: {e}")
            self.model_path = model_path or "C:/devhome/projects/models/meta-llama-Llama-3-2-1B-Instruct/meta-llama-Llama-3-2-1B-Instruct"
            self.max_tokens = 512
            self.temperature = 0.7
            self.top_p = 0.9
            self.repetition_penalty = 1.1
        
        self.tokenizer = None
        self.model = None
        self.device = None
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load the LLaMA model and tokenizer."""
        try:
            logger.info(f"Loading LLaMA model from: {self.model_path}")
            
            # Check if model path exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model path does not exist: {self.model_path}")
                return False
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                local_files_only=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                local_files_only=True, 
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                self.model.to(self.device)
            
            self.model_loaded = True
            logger.info(f"✅ LLaMA model loaded successfully on {self.device.upper()}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading LLaMA model: {str(e)}")
            self.model_loaded = False
            return False
    
    def _generate_response(self, prompt: str, max_tokens: int = None, temperature: float = None, top_p: float = None) -> str:
        """Generate a response using the LLaMA model."""
        if not self.model_loaded:
            return "❌ Model not loaded. Please check the model path."
        
        # Use instance defaults if not provided
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        
        try:
            # Format input for LLaMA 3.2 Instruct
            formatted_input = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_input, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=self.repetition_penalty
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response
            if formatted_input in response:
                response = response.replace(formatted_input, "").strip()
            
            response = response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
            
            if not response:
                response = "I understand your question, but I'm having trouble generating a response right now. Could you please rephrase your question?"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"❌ Error generating response: {str(e)}"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResult:
        """Chat completion method for Crew AI integration."""
        if not self.model_loaded:
            return LLMResult(
                content="❌ Model not loaded. Please check the model path.",
                error="Model not loaded"
            )
        
        try:
            # Extract the last user message for simplicity
            # In a more sophisticated implementation, you might want to handle conversation history
            user_message = None
            for message in reversed(messages):
                if message.get("role") == "user":
                    user_message = message.get("content", "")
                    break
            
            if not user_message:
                return LLMResult(
                    content="❌ No user message found in conversation.",
                    error="No user message"
                )
            
            # Generate response
            response = self._generate_response(
                user_message,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p)
            )
            
            return LLMResult(content=response)
            
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            return LLMResult(
                content=f"❌ Error in chat completion: {str(e)}",
                error=str(e)
            )
    
    def complete(self, prompt: str, **kwargs) -> LLMResult:
        """Text completion method for Crew AI integration."""
        if not self.model_loaded:
            return LLMResult(
                content="❌ Model not loaded. Please check the model path.",
                error="Model not loaded"
            )
        
        try:
            response = self._generate_response(
                prompt,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p)
            )
            
            return LLMResult(content=response)
            
        except Exception as e:
            logger.error(f"Error in text completion: {str(e)}")
            return LLMResult(
                content=f"❌ Error in text completion: {str(e)}",
                error=str(e)
            )
    
    def is_model_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": self.model_path,
            "device": self.device,
            "model_type": type(self.model).__name__,
            "tokenizer_type": type(self.tokenizer).__name__,
            "cuda_available": torch.cuda.is_available(),
            "config": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty
            }
        }
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.model_loaded = False
        logger.info("Model resources cleaned up")

# Global instance for use across the application
local_llm = LocalLLaMALLM()
