from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ocr_url: str = "https://qwen.aigroupdev.com/v1/ocr"
    ocr_token: str = ""
    ocr_timeout: float = 120.0
    ocr_verify_ssl: bool = True
    ocr_file_field: str = "file"

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
