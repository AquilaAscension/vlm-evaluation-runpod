import os, base64, requests, json, pathlib
class AnthropicVLM:
    def __init__(self, model_id="claude-3-sonnet-20240229"):
        self.key = os.environ["ANTHROPIC_API_KEY"]
        self.model = model_id
    def generate(self, image_path, prompt):
        img_b64 = base64.b64encode(open(image_path, "rb").read()).decode()
        payload = {
          "model": self.model,
          "max_tokens": 64,
          "messages":[{
            "role":"user",
            "content":[
              {"type":"image", "source":{"type":"base64","data":img_b64,"media_type":"image/jpeg"}},
              {"type":"text", "text": prompt}
            ]}]
        }
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            json=payload,
            timeout=60,
        )
        return r.json()["content"][0]["text"].strip()
