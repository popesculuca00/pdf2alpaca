import time
import random
from openai import OpenAI
import streamlit as st

get_client = lambda port_num: OpenAI(
    api_key="EMPTY",
    base_url=f"http://localhost:{port_num}/v1",
)


def get_clients():
    try:
        gemma_client = get_client(8000)
    except:
        gemma_client = None

    try:
        qwen_client = get_client(8001)
    except:
        qwen_client = None

    return{
        "google/gemma-3-1b-it": gemma_client,
        "Qwen/Qwen2.5-1.5B-Instruct": qwen_client,
    }

st.session_state["clients"] = get_clients()


class HRLLMModel:
    def __init__(self):
        self.model_name = None
        self.current_response = None
        self._available_models = None

    def select_model(self, model_name):
        """Set the model to use for generation"""
        if model_name:
            self.model_name = model_name

    def generate_response(self, query):
        """Generate a response for the given query"""
        if not self.model_name:
            available_models = self.get_available_models()
            if available_models:
                self.model_name = available_models[0]
        
        if not self.model_name:
            self.model_name = "llama3"
        
        try:
            
            system_prompt = f"You are a helpful HR assistant. Be concise and friendly in your responses."
            
            self.current_response = st.session_state["clients"][self.model_name].chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                model=self.model_name,
                stream=True
            )
            
            for chunk in self.current_response:
                if "is_generating" in st.session_state and not st.session_state.is_generating:
                    try:
                        self.current_response.close()
                    except:
                        pass
                    break
                    
                content = chunk.choices[0].delta.content if chunk.choices and hasattr(chunk.choices[0].delta, 'content') else ""
                if content is not None:
                    yield content
                
        except Exception as e:
            yield f"Error: Unable to generate response with model '{self.model_name}'. Details: {str(e)}"
            if "is_generating" in st.session_state:
                st.session_state.is_generating = False

    def get_available_models(self):
        """Get a list of available models"""
        
        st.session_state["clients"] = get_clients()
        return list(st.session_state["clients"].keys())