
import os
from openai import OpenAI
from src.utils.config import get_config


def get_client():
    config = get_config()

    base_url = config["api"]["base_url"]
    api_key = os.getenv("MYAPIKEY1")

    if base_url:
        print(f"Using custom base_url: {base_url}")
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=120.0,
        )

    print("Using OpenAI API")
    return OpenAI(
        api_key=api_key,
        timeout=120.0,
    )

