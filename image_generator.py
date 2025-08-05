import requests
import json
import base64
from io import BytesIO
from PIL import Image
import streamlit as st

class ImageGenerator:
    def __init__(self):
        # Using Hugging Face's free API for image generation
        self.api_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        # You can get a free API key from huggingface.co
        self.headers = {"Authorization": "Bearer hf_your_token_here"}  # Replace with your token
        
    def generate_image_pollinations(self, prompt):
        """Generate image using Pollinations AI (free service)"""
        try:
            # Pollinations AI free API
            url = f"https://image.pollinations.ai/prompt/{prompt}"
            
            # Add parameters for better quality
            params = {
                'width': '512',
                'height': '512',
                'seed': '-1',
                'nologo': 'true'
            }
            
            # Make request with parameters
            full_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
            
            return full_url
            
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            return None
    
    def generate_image_huggingface(self, prompt):
        """Generate image using Hugging Face API (requires token)"""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": "blurry, bad quality, distorted",
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5,
                    "width": 512,
                    "height": 512
                }
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                image_bytes = response.content
                image = Image.open(BytesIO(image_bytes))
                
                # Convert to base64 for display
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                return f"data:image/png;base64,{image_base64}"
            else:
                st.error(f"API Error: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            return None
    
    def generate_image_picsum(self, prompt):
        """Generate placeholder image using Lorem Picsum (for demo purposes)"""
        try:
            # This creates a placeholder image - you can enhance this with actual AI generation
            import hashlib
            
            # Create a seed from the prompt
            seed = abs(hash(prompt)) % 1000
            
            # Lorem Picsum with seed for consistent images
            url = f"https://picsum.photos/seed/{seed}/512/512"
            
            return url
            
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            return None
    
    def generate_image(self, prompt):
        """Main method to generate image - tries multiple services"""
        
        # First try Pollinations AI (free)
        try:
            image_url = self.generate_image_pollinations(prompt)
            if image_url:
                return image_url
        except:
            pass
        
        # Fallback to placeholder service
        try:
            return self.generate_image_picsum(prompt)
        except:
            return None
    
    def enhance_prompt(self, basic_prompt):
        """Enhance the prompt for better image generation"""
        enhancements = [
            "high quality",
            "detailed",
            "professional",
            "clear",
            "well-lit",
            "sharp focus"
        ]
        
        enhanced = f"{basic_prompt}, {', '.join(enhancements)}"
        return enhanced
    
    def generate_academic_image(self, subject, style="educational"):
        """Generate subject-specific academic images"""
        academic_prompts = {
            "mathematics": "mathematical equations, graphs, geometric shapes, academic setting",
            "science": "laboratory equipment, scientific diagrams, research setting",
            "literature": "books, writing materials, literary atmosphere",
            "history": "historical documents, timeline, academic research",
            "general": "study materials, academic environment, educational setting"
        }
        
        base_prompt = academic_prompts.get(subject.lower(), academic_prompts["general"])
        
        if style == "diagram":
            prompt = f"educational diagram of {subject}, clear labels, informative, professional"
        elif style == "infographic":
            prompt = f"infographic about {subject}, colorful, educational, easy to understand"
        else:
            prompt = f"{base_prompt}, educational illustration"
        
        return self.generate_image(self.enhance_prompt(prompt))

# Example usage and testing functions
def test_image_generator():
    """Test function for the image generator"""
    generator = ImageGenerator()
    
    # Test basic generation
    test_prompt = "a beautiful sunset over mountains"
    result = generator.generate_image(test_prompt)
    
    if result:
        print(f"Image generated successfully: {result}")
    else:
        print("Failed to generate image")
    
    # Test academic image generation
    academic_result = generator.generate_academic_image("mathematics", "diagram")
    if academic_result:
        print(f"Academic image generated: {academic_result}")

if __name__ == "__main__":
    test_image_generator()