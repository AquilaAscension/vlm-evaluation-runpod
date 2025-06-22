import os, base64, requests, json, pathlib

class AnthropicVLM:
    def __init__(self, model_id="claude-4-sonnet-20250522"):
        self.key = os.environ["ANTHROPIC_API_KEY"]
        self.model = model_id

    def generate(self, image_path, prompt):
        img_b64 = base64.b64encode(open(image_path, "rb").read()).decode()
        payload = {
          "model": self.model,
          "max_tokens": 64,
          "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"<image:{img_b64}> {prompt}"}
          ]
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        # Extract assistant's reply text
        return result["completion"]

    # (Any additional methods unchanged)
