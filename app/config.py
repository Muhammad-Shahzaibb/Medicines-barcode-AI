from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI-compatible vision endpoint
    ocr_url: str = "https://101.46.52.232/v2/infer/de9b8e84-0f1b-472a-bc30-27b5eab6fec6/v1/chat/completions"
    ocr_token: str = ""
    ocr_model: str = "qwen3.8"
    ocr_max_tokens: int = 1024
    ocr_timeout: float = 120.0
    ocr_verify_ssl: bool = False  # self-signed cert on the inference server

    max_concurrency: int = 4
    rotation_search: bool = True
    rotation_angles: str = "0,90,270,180"
    max_image_side: int = 2200

    @property
    def angles(self) -> list[int]:
        out = []
        for a in self.rotation_angles.split(","):
            a = a.strip()
            if a:
                out.append(int(a) % 360)
        return out or [0]


@lru_cache
def get_settings() -> Settings:
    return Settings()
