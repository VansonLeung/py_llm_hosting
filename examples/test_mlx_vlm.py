#!/usr/bin/env python3
"""Quick tester for MLX-VLM chat completions with image inputs."""

import argparse
import base64
from pathlib import Path
import sys

import httpx


def build_data_uri(image_path: Path) -> str:
    if not image_path.exists():
        raise SystemExit(f"Image file not found: {image_path}")
    mime = "image/png"
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif image_path.suffix.lower() == ".gif":
        mime = "image/gif"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a multimodal chat request to the local API")
    parser.add_argument("--model", default="chandra-4bit", help="Model name configured in servers.json")
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--prompt", default="Describe this image", help="User prompt")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    args = parser.parse_args()

    image_uri = build_data_uri(Path(args.image))

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": image_uri}}
                ]
            }
        ]
    }

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    with httpx.Client(timeout=180.0) as client:
        response = client.post(url, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - surfacing error is enough
            print(f"Request failed: {exc.response.text}", file=sys.stderr)
            raise

    data = response.json()
    message = data["choices"][0]["message"]["content"]
    print("\nAssistant Response:\n-------------------")
    print(message)


+if __name__ == "__main__":
+    main()
