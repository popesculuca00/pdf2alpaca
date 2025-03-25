import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

class LlamaInference:
    """
    A class for performing inference with a fine-tuned Llama model with token streaming.
    """
    
    def __init__(self, model_path, load_in_4bit=True, device="auto"):
        """
        Initialize the model and tokenizer.
        
        Args:
            model_path (str): Path to the fine-tuned model
            load_in_4bit (bool): Whether to load the model in 4-bit precision
            device (str): Device to place the model on ("auto", "cuda", "cpu")
        """
        self.model_path = model_path
        self.load_model(load_in_4bit, device)
    
    def load_model(self, load_in_4bit=True, device="auto"):
        """
        Load the fine-tuned model and tokenizer.
        """
        print(f"Loading model from {self.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device,
            torch_dtype=torch.float16,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
    
    def format_prompt(self, user_input):
        """
        Format the prompt according to Llama 3.2 chat template.
        
        Args:
            user_input (str): User's input question or prompt
        
        Returns:
            str: Formatted prompt ready for the model
        """
        return f"<|begin_of_text|><|user|>\n{user_input}<|end_of_user|>\n<|assistant|>\n"
    
    def generate_response(self, question, max_new_tokens=2048, temperature=0.1, top_p=0.9, *args, **kwargs):
        """
        Generate a response to the input question and yield tokens as they're generated.
        
        Args:
            question (str): Input question or prompt
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (lower = more deterministic)
            top_p (float): Nucleus sampling parameter
            
        Yields:
            str: Text tokens as they are generated
        """

        prompt = self.format_prompt(question)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_special_tokens=True, 
            skip_prompt=True
        )

        generation_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_token in streamer:
            yield new_token

    def select_model(self, *args, **kwargs):
        pass

    def get_available_models(self):
        """Get a list of available models"""
        return ["llama3.1-finetuned"]