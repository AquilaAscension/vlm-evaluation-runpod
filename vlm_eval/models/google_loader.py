import os, base64, google.generativeai as genai

class GeminiVLM:
    """Gemini 2.5 Flash via Google Generative AI SDK."""

    def __init__(self, model_id: str = "gemini-2.5-flash"):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model_id)

    def generate(self, image_path: str, prompt: str, **_) -> str:
        img_bytes = open(image_path, "rb").read()
        r = self.model.generate_content(
            [ {"mime_type":"image/jpeg", "data": img_bytes}, prompt ],
            max_output_tokens=64
        )
        return r.text.strip()
