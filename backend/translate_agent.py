# translate_agent.py (original version with same-lang + pivot)
import os
import logging
from functools import lru_cache
from typing import List, Union, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from transformers import pipeline
from huggingface_hub.errors import RepositoryNotFoundError

# Optional language detection
DETECT_ENABLED = True
try:
    from langdetect import detect as _detect_lang
except Exception:
    DETECT_ENABLED = False
    _detect_lang = None

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("translate.agent")

device = 0 if torch.cuda.is_available() else -1
logger.info("Translate agent device: %s", "cuda" if device == 0 else "cpu")

# OPUS-MT language aliases
LANG_ALIASES = {
    "en": "en","fr": "fr","es": "es","de": "de","it": "it","pt": "pt","nl": "nl",
    "sv": "sv","no": "no","da": "da","fi": "fi","ru": "ru","uk": "uk","pl": "pl",
    "cs": "cs","sk": "sk","ro": "ro","bg": "bg","el": "el","tr": "tr","ar": "ar",
    "fa": "fa","he": "he","hi": "hi","bn": "bn","ta": "ta","te": "te","zh": "zh",
    "ja": "ja","ko": "ko",
}

def _norm(code: str) -> str:
    return LANG_ALIASES.get(code.lower(), code.lower())

def _pair(src: str, tgt: str) -> str:
    return f"{_norm(src)}-{_norm(tgt)}"

@lru_cache(maxsize=32)
def get_translator(src: str, tgt: str):
    model_id = f"Helsinki-NLP/opus-mt-{_pair(src, tgt)}"
    logger.info("Loading translation model: %s", model_id)
    try:
        return pipeline("translation", model=model_id, device=device)
    except RepositoryNotFoundError as e:
        raise e

app = FastAPI(title="Translation Agent", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateRequest(BaseModel):
    text: Union[str, List[str]]
    tgt_lang: str
    src_lang: Optional[str] = None
    max_length: Optional[int] = 512

class TranslateResponse(BaseModel):
    translated_text: Union[str, List[str]]
    src_lang: str
    tgt_lang: str
    pivoted: bool = False

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/languages")
def languages():
    return {"languages": sorted(LANG_ALIASES.keys())}

def _detect(text: str) -> str:
    if not DETECT_ENABLED or not _detect_lang:
        return "en"
    try:
        return _norm(_detect_lang(text))
    except Exception:
        return "en"

def _ensure_list(x):
    return x if isinstance(x, list) else [x]

def _maybe_shorten(texts: List[str]) -> List[str]:
    return [t if len(t) < 8000 else t[:8000] for t in texts]

@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    texts = _ensure_list(req.text)
    texts = _maybe_shorten(texts)

    src = _norm((req.src_lang or "").strip()) if req.src_lang else ""
    if src in ("", "auto", None):
        sample = next((t for t in texts if t and t.strip()), "")
        src = _detect(sample) if sample else "en"

    tgt = _norm(req.tgt_lang)
    max_len = req.max_length or 512

    # same-language shortcut
    if src == tgt:
        return TranslateResponse(
            translated_text=req.text,
            src_lang=src,
            tgt_lang=tgt,
            pivoted=False,
        )

    def hop(_src, _tgt, _texts):
        translator = get_translator(_src, _tgt)
        outs = translator(_texts, max_length=max_len, clean_up_tokenization_spaces=True)
        return [o["translation_text"] for o in outs]

    try:
        out = hop(src, tgt, texts)
        return TranslateResponse(
            translated_text=out if isinstance(req.text, list) else out[0],
            src_lang=src,
            tgt_lang=tgt,
            pivoted=False,
        )
    except RepositoryNotFoundError:
        if src != "en" and tgt != "en":
            try:
                mid = hop(src, "en", texts)
                out = hop("en", tgt, mid)
                return TranslateResponse(
                    translated_text=out if isinstance(req.text, list) else out[0],
                    src_lang=src,
                    tgt_lang=tgt,
                    pivoted=True,
                )
            except RepositoryNotFoundError:
                pass
        raise HTTPException(
            status_code=400,
            detail={
                "error": "no_model_for_language_pair",
                "message": f"No OPUS-MT model for '{src}->{tgt}'. Try another target language.",
                "src_lang": src,
                "tgt_lang": tgt,
            },
        )
    except Exception as e:
        logger.exception("Unexpected translate error: %s", e)
        raise HTTPException(status_code=500, detail="internal_error")
def translate_core(req: TranslateRequest) -> TranslateResponse:
    texts = req.text if isinstance(req.text, list) else [req.text]
    texts = [t if len(t) < 8000 else t[:8000] for t in texts]

    # detect src if missing
    src = _norm(req.src_lang) if req.src_lang not in ("", None, "auto") else ""
    if not src:
        sample = next((t for t in texts if t and t.strip()), "")
        src = _detect(sample) if sample else "en"

    tgt = _norm(req.tgt_lang)
    max_len = req.max_length or 512

    # same-lang shortcut
    if src == tgt:
        return TranslateResponse(
            translated_text=req.text,
            src_lang=src,
            tgt_lang=tgt,
            pivoted=False,
        )

    def hop(_src, _tgt, _texts):
        translator = get_translator(_src, _tgt)
        outs = translator(_texts, max_length=max_len, clean_up_tokenization_spaces=True)
        return [o["translation_text"] for o in outs]

    try:
        out = hop(src, tgt, texts)
        return TranslateResponse(
            translated_text=out if isinstance(req.text, list) else out[0],
            src_lang=src,
            tgt_lang=tgt,
            pivoted=False,
        )
    except RepositoryNotFoundError:
        if src != "en" and tgt != "en":
            mid = hop(src, "en", texts)
            out = hop("en", tgt, mid)
            return TranslateResponse(
                translated_text=out if isinstance(req.text, list) else out[0],
                src_lang=src,
                tgt_lang=tgt,
                pivoted=True,
            )
        raise HTTPException(status_code=400, detail="no_model_for_language_pair")
