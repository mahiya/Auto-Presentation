import os
import base64
import hashlib
from typing import List, Optional, Any
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF

try:
    from openai import AzureOpenAI
except ImportError:
    st.error("`openai` パッケージが見つかりません。`pip install openai` を実行してください。")
    st.stop()
import streamlit.components.v1 as components
import re
import collections
import requests
from urllib.parse import urljoin, urlparse

# ─────────────────────────────────────────────────────────────────────────────
# Environment Initialization
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOY_SUMMARY = os.getenv("AZURE_GPT5_DEPLOYMENT")
DEPLOY_TTS = os.getenv("AZURE_TTS_DEPLOYMENT")
DEPLOY_QA = os.getenv("AZURE_GPT5_QA_DEPLOYMENT") or DEPLOY_SUMMARY  # 専用が無ければ要約モデル再利用

# 固定値の定義
TTS_SPEED = 3.5
TTS_VOICE = "alloy"  # サポートされているボイス名に変更
DEFAULT_AUDIO_FORMAT = "mp3"

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    st.warning("⚠️ .env に Azure OpenAI の接続情報が設定されていません。サンプルに従って設定してください。")


# Initialize Azure client
def get_azure_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Auto Slide Summarizer + TTS", layout="wide")
st.title("AI Auto Presentation")
st.caption("AIが自動でプレゼンテーション、質疑応答を実施します")

# 言語選択（アップローダ直下）
lang_label = st.radio("言語 / Language", ["日本語", "English"], horizontal=True, key="lang_selector")
ui_lang = "ja" if lang_label == "日本語" else "en"

# 言語変更検知とキャッシュクリア
if "prev_lang" not in st.session_state:
    st.session_state.prev_lang = ui_lang
if st.session_state.prev_lang != ui_lang:
    st.session_state.summary_cache.clear()
    st.session_state.audio_cache.clear()
    st.session_state.summary_audio_played_pages.clear()
    st.session_state.answer_cache.clear()
    st.session_state.answer_audio_cache.clear()
    st.session_state.prev_lang = ui_lang
st.session_state["ui_lang"] = ui_lang

# 言語別 TTS ボイス
if ui_lang == "en":
    voice = "alloy"
else:
    voice = TTS_VOICE

# レイアウトCSSは不要（デフォルトwideレイアウト利用）


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def file_bytes_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def render_pdf_page_pixmap(page, scale: float = 2.0) -> bytes:
    """Render a PDF page to PNG bytes. scale=2.0 is a good balance."""
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def encode_b64_png(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")


def extract_page_text(page) -> str:
    # Basic text extraction; you can switch to blocks/dict for layout-aware summaries
    return page.get_text("text", sort=True) or ""


def _responses_content_for(page_text: str, page_image_b64: Optional[str], language: str) -> list:
    if language == "en":
        instructions = (
            "You are a professional presenter. Based on the PDF page content, craft a spoken-style English presentation script. "
            "Make it engaging, naturally including contextual introduction, background, examples, key point explanation, and a brief wrap‑up, "
            "but do NOT label those sections explicitly. Length: about 120–200 English words. "
            "Do not add greetings like 'Hello everyone'. Provide a natural flow as if you are speaking."
        )
    else:
        instructions = (
            "あなたはプロのプレゼンテーターです。以下のPDFページ内容をもとに、聴衆の興味を引く日本語のプレゼンテーション原稿を作成してください。"
            "話し言葉で、導入・背景説明・具体例・ポイント解説・まとめなどを自然に含めますが、各要素名は明示しないでください。"
            "長さは120〜200語相当（日本語で200〜400文字程度）を目安に自然な流れで説明してください。"
            "挨拶（例: みなさんこんにちは 等）は入れないでください。"
        )
    content = [{"type": "input_text", "text": instructions}]
    clipped = (page_text or "")[:2000]
    if clipped.strip():
        label = "Page extracted text (excerpt)" if language == "en" else "【ページ抽出テキスト（抜粋）】"
        content.append({"type": "input_text", "text": f"{label}\n{clipped}"})
    if page_image_b64:
        content.append({"type": "input_image", "image_url": {"url": f"data:image/png;base64,{page_image_b64}"}})
    return content


def _chat_content_for(page_text: str, page_image_b64: Optional[str], language: str) -> list:
    if language == "en":
        base = (
            "You are a professional presenter. Based on the PDF page, write an engaging, spoken-style English presentation script. "
            "Include an organic intro, background, examples, key point explanation, and a short wrap‑up without labeling sections explicitly. "
            "Around 120–200 English words. Avoid greetings."
        )
        excerpt_label = "Page extracted text (excerpt)"
    else:
        base = (
            "あなたはプロのプレゼンテーターです。PDFページ内容を基に、話し言葉で自然な日本語プレゼン原稿を作成してください。"
            "導入・背景・具体例・要点・簡潔な締めを自然に含め、セクション名は明示しないでください。挨拶は不要。"
            "目安 200〜400文字。"
        )
        excerpt_label = "【ページ抽出テキスト（抜粋）】"
    user_content = [{"type": "text", "text": base}]
    clipped = (page_text or "")[:800]
    if clipped.strip():
        user_content.append({"type": "text", "text": f"{excerpt_label}\n{clipped}"})
    if page_image_b64:
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_image_b64}"}})
    return user_content


def extract_text_from_response(resp: Any) -> str:
    """
    Try to extract text robustly from either Responses API result or Chat Completions result.
    """

    def _collect_text(content: Any) -> List[str]:
        parts: List[str] = []
        if content is None:
            return parts
        if isinstance(content, str):
            parts.append(content)
            return parts
        if isinstance(content, (list, tuple)):
            for item in content:
                parts.extend(_collect_text(item))
            return parts
        if isinstance(content, dict):
            text_val = content.get("text")
            if isinstance(text_val, str) and text_val:
                parts.append(text_val)
            elif text_val is not None:
                parts.extend(_collect_text(text_val))
            for key in ("content", "message", "value"):
                if key in content:
                    parts.extend(_collect_text(content[key]))
            return parts
        text_attr = getattr(content, "text", None)
        if isinstance(text_attr, str) and text_attr:
            parts.append(text_attr)
        elif text_attr is not None:
            parts.extend(_collect_text(text_attr))
        content_attr = getattr(content, "content", None)
        if content_attr is not None:
            parts.extend(_collect_text(content_attr))
        return parts

    # 1) Responses API convenience
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
        return resp.output_text.strip()

    # 2) Responses API structured content
    try:
        # pydantic model to dict
        data = resp.model_dump() if hasattr(resp, "model_dump") else resp
        if isinstance(data, dict) and "output" in data:
            pieces: List[str] = []
            for item in data.get("output", []):
                pieces.extend(_collect_text(item))
            if pieces:
                return "\n".join(pieces).strip()
    except Exception:
        pass

    # 3) Chat Completions API
    try:
        if hasattr(resp, "choices") and resp.choices:
            msg = resp.choices[0].message
            if msg:
                msg_content = getattr(msg, "content", None)
                pieces = _collect_text(msg_content)
                if pieces:
                    return "\n".join(piece.strip() for piece in pieces if piece.strip()).strip()
                msg_text = getattr(msg, "text", None)
                if isinstance(msg_text, str) and msg_text.strip():
                    return msg_text.strip()
        # legacy text attribute on choice directly
        if hasattr(resp, "choices") and resp.choices:
            choice = resp.choices[0]
            choice_text = getattr(choice, "text", None)
            if isinstance(choice_text, str) and choice_text.strip():
                return choice_text.strip()
    except Exception:
        pass

    # 4) Fallback to string conversion
    return (str(resp) if resp is not None else "").strip()


def build_multimodal_summary(
    client: AzureOpenAI,
    deployment_name: str,
    page_text: str,
    page_image_b64: Optional[str],
    max_output_tokens: int = 500,
    language: str = "ja",
) -> str:
    """
    Prefer Responses API then fallback to Chat Completions.
    （temperature パラメータは削除）
    """
    if language == "en":
        system_prompt = "You are a skilled presentation assistant. Produce a concise spoken-style explanation in English highlighting key points."
    else:
        system_prompt = "あなたは有能な要約アシスタントです。短く要点を外さずに説明します。"

    def _call_responses():
        return client.responses.create(
            model=deployment_name,
            input=[
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": _responses_content_for(page_text, page_image_b64, language)},
            ],
            max_output_tokens=max_output_tokens,
        )

    # Responses API
    try:
        resp = _call_responses()
        txt = extract_text_from_response(resp)
        if txt:
            return txt
    except Exception:
        pass

    # Chat Completions fallback
    try:
        resp = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _chat_content_for(page_text, page_image_b64, language)},
            ],
            max_completion_tokens=max_output_tokens,
        )
        txt = extract_text_from_response(resp)
        if txt:
            return txt
    except Exception:
        pass

    return "（このページの要約を生成できませんでした）"


def synthesize_tts(
    client: AzureOpenAI,
    deployment_name: str,
    text: str,
    voice: str = "alloy",
    audio_format: str = "mp3",  # "mp3" | "wav" | "ogg"
) -> bytes:
    """
    Azure OpenAI gpt-4o-mini-tts を audio.speech エンドポイントで呼び出して音声を生成。
    戻り値: 音声バイナリ。
    """
    response_format = "opus" if audio_format == "ogg" else audio_format

    # Non-streaming (fastest path if supported by SDK)
    try:
        resp = client.audio.speech.create(
            model=deployment_name,  # Azure: deployment name
            voice=voice,  # alloy / echo / fable / onyx / nova / shimmer ...
            input=text,
            response_format=response_format,  # important: response_format
        )
        if hasattr(resp, "read") and callable(resp.read):
            return resp.read()
        if hasattr(resp, "content"):
            return resp.content  # type: ignore[attr-defined]
        if isinstance(resp, (bytes, bytearray)):
            return bytes(resp)
    except Exception:
        pass  # fallback to streaming

    # Streaming fallback (compat)
    import tempfile

    with client.audio.speech.with_streaming_response.create(
        model=deployment_name,
        voice=voice,
        input=text,
        response_format=response_format,
    ) as response:
        suffix = ".ogg" if audio_format == "ogg" else f".{audio_format}"
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            response.stream_to_file(tmp.name)
            tmp.seek(0)
            return tmp.read()


def estimate_speech_seconds(text: str, chars_per_sec: float = 7.0, min_sec: float = 3.0) -> float:
    """Rough estimate for JP speech length. Adjust via UI."""
    n = len(text or "")
    est = n / max(chars_per_sec, 1.0)
    return max(est, min_sec)


def render_inline_audio(audio_bytes: bytes, fmt: str, autoplay: bool, height: int = 70):
    if not audio_bytes:
        return
    try:
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        mime = "audio/ogg" if fmt == "ogg" else f"audio/{fmt}"
        auto_attr = "autoplay " if autoplay else ""
        html = f"<audio {auto_attr}controls src='data:{mime};base64,{b64}'></audio>"
        components.html(html, height=height)
    except Exception as e:
        st.error(f"回答音声プレーヤー生成エラー: {e}")


def clean_html(raw: str) -> str:
    if not raw:
        return ""
    text = raw
    try:
        from bs4 import BeautifulSoup  # optional

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    except Exception:
        text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
        text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_chunks(text: str, size: int = 800, overlap: int = 100) -> list:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + size
        chunks.append(text[start:end])
        if end >= length:
            break
        start = end - overlap
    return chunks


def crawl_site(base_url: str, max_pages: int = 5, timeout: int = 8) -> str:
    if not base_url:
        return ""
    seen = set()
    out_texts = []
    queue = collections.deque([base_url])
    domain = urlparse(base_url).netloc
    while queue and len(seen) < max_pages:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200 or "text" not in r.headers.get("Content-Type", ""):
                continue
            html = r.text
            text = clean_html(html)
            if text:
                out_texts.append(text)
            # 簡易リンク抽出
            for m in re.findall(r'href=["\'](.*?)["\']', html, flags=re.I):
                if m.startswith("mailto:") or m.startswith("javascript:"):
                    continue
                new_url = urljoin(url, m)
                if urlparse(new_url).netloc == domain and len(seen) + len(queue) < max_pages:
                    queue.append(new_url)
        except Exception:
            continue
    return "\n".join(out_texts)


def simple_retrieval(question: str, chunks: list, top_k: int = 5) -> list:
    if not question or not chunks:
        return []
    q_terms = [w for w in re.split(r"\W+", question.lower()) if w]
    scored = []
    for c in chunks:
        lc = c.lower()
        score = sum(lc.count(t) for t in q_terms if t)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored[:top_k] if s > 0]


def expand_question_terms(question: str) -> List[str]:
    """簡易的に質問語を拡張（英語/日本語の基本同義語）"""
    base = [w for w in re.split(r"\W+", (question or "").lower()) if w]
    extra = []
    syn_map = {
        "benefit": ["advantage", "value"],
        "advantage": ["benefit"],
        "challenge": ["problem", "課題"],
        "課題": ["問題", "challenge"],
        "purpose": ["goal", "目的"],
        "目的": ["狙い", "ゴール"],
        "背景": ["背景情報", "background"],
        "summary": ["overview"],
        "概要": ["サマリ", "まとめ"],
    }
    for w in base:
        extra += syn_map.get(w, [])
    # 日本語ひら/カナ簡易揺れ
    if any("ポイント" in q for q in base):
        extra.append("要点")
    if any("要点" in q for q in base):
        extra.append("ポイント")
    return list(dict.fromkeys(base + extra))


def scored_chunks(question: str, chunks: List[str], top_k: int = 5) -> List[str]:
    terms = expand_question_terms(question)
    if not terms or not chunks:
        return []
    scored = []
    for c in chunks:
        lc = c.lower()
        score = 0
        for t in terms:
            if not t:
                continue
            score += lc.count(t.lower())
        # ボーナス: 先頭出現
        for t in terms[:3]:
            if lc.startswith(t.lower()):
                score += 2
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored[:top_k]]


# build_slide_qa_answer をシンプル版（公開情報のみ）+ パラメータ互換フォールバック
def build_slide_qa_answer(
    client: AzureOpenAI,
    deployment_name: str,
    slide_text: str,
    question: str,
    image_b64: Optional[str],
    full_pdf_chunks: list,
    site_chunks: list,
    language: str = "ja",
    max_output_tokens: int = 512,
) -> str:
    q = (question or "").strip()
    if not q:
        return "（質問が空です）"

    if language == "en":
        system_txt = (
            "You are a concise public-general-knowledge Q&A assistant. Answer about 120-200 English words."
            "Do NOT spend many tokens thinking silently; output the answer succinctly."
        )
        user_txt = f"Question:\n{q}\nProvide a direct answer."
        empty_ans = "(No answer)"
    else:
        system_txt = (
            "あなたは一般公開知識で簡潔に答えるQ&Aアシスタントです。200~400文字程度で回答してください。"
            "本当に不確かな時だけ『不確かです』と述べます。"
        )
        user_txt = f"質問:\n{q}\n端的に回答してください。"
        empty_ans = "（回答なし）"

    def _extract_any(resp) -> str:
        try:
            txt = extract_text_from_response(resp)
            if txt and txt.strip():
                return txt.strip()
        except Exception:
            pass
        # 直接 choices 参照
        try:
            if hasattr(resp, "choices") and resp.choices:
                ch = resp.choices[0]
                msg = getattr(ch, "message", None)
                if msg:
                    mc = getattr(msg, "content", "")

                    def _fallback_content_to_text(content: Any) -> str:
                        if content is None:
                            return ""
                        if isinstance(content, str):
                            return content.strip()
                        if isinstance(content, (list, tuple)):
                            parts = [_fallback_content_to_text(item) for item in content]
                            return "\n".join(part for part in parts if part).strip()
                        if isinstance(content, dict):
                            text_val = content.get("text")
                            if isinstance(text_val, str) and text_val.strip():
                                return text_val.strip()
                            gathered = []
                            if text_val is not None:
                                gathered.append(_fallback_content_to_text(text_val))
                            for key in ("content", "value"):
                                if key in content:
                                    gathered.append(_fallback_content_to_text(content.get(key)))
                            return "\n".join(val for val in gathered if val).strip()
                        text_attr = getattr(content, "text", None)
                        if isinstance(text_attr, str) and text_attr.strip():
                            return text_attr.strip()
                        if text_attr is not None:
                            return _fallback_content_to_text(text_attr)
                        content_attr = getattr(content, "content", None)
                        if content_attr is not None:
                            return _fallback_content_to_text(content_attr)
                        return ""

                    if isinstance(mc, (list, tuple, dict)):
                        txt = _fallback_content_to_text(mc)
                    elif isinstance(mc, str):
                        txt = mc.strip()
                    if txt:
                        return txt
                if hasattr(ch, "text") and ch.text:
                    return ch.text.strip()
        except Exception:
            pass
        return ""

    debug_logs = st.session_state.get("qa_debug_logs")

    # 1st attempt (Chat Completions)
    messages = [
        {"role": "system", "content": system_txt},
        {"role": "user", "content": user_txt},
    ]
    try:
        resp = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
        )
        ans = _extract_any(resp)
        if ans:
            return ans
        # 判定: reasoning_tokens が上限近く
        reasoning_tokens = getattr(getattr(resp, "usage", None), "completion_tokens_details", None)
        r_used = 0
        if reasoning_tokens:
            r_used = getattr(reasoning_tokens, "reasoning_tokens", 0)
        finish_reason = ""
        if hasattr(resp, "choices") and resp.choices:
            finish_reason = getattr(resp.choices[0], "finish_reason", "") or getattr(resp.choices[0], "finishReason", "")
        need_retry_reasoning = (finish_reason == "length") and (r_used >= max_output_tokens * 0.8)

        if debug_logs is not None:
            debug_logs.append(f"Q1 empty finish={finish_reason} reasoning_used={r_used}")

        if not need_retry_reasoning:
            # content_filter?
            if finish_reason and "content_filter" in finish_reason.lower():
                return "（コンテンツフィルターにより回答が生成されませんでした）"
        else:
            # 2nd attempt: increase token budget + 強い即答指示
            if language == "en":
                system_txt_retry = system_txt + " Produce the final answer within 2 short sentences. Do not over-think."
            else:
                system_txt_retry = system_txt + " 2文以内で即答してください。長い思考は不要。"
            messages_retry = [
                {"role": "system", "content": system_txt_retry},
                {"role": "user", "content": user_txt},
            ]
            try:
                resp2 = client.chat.completions.create(
                    model=deployment_name,
                    messages=messages_retry,
                )
                ans2 = _extract_any(resp2)
                if ans2:
                    return ans2
                if debug_logs is not None:
                    fr2 = ""
                    if hasattr(resp2, "choices") and resp2.choices:
                        fr2 = getattr(resp2.choices[0], "finish_reason", "") or getattr(resp2.choices[0], "finishReason", "")
                    debug_logs.append(f"Q2 empty finish={fr2}")
            except Exception as e2:
                if debug_logs is not None:
                    debug_logs.append(f"Q2 exception: {e2}")
    except Exception as e:
        return f"（回答生成でエラー: {e}）"

    # 3rd attempt: Responses API fallback
    try:
        resp_r = client.responses.create(
            model=deployment_name,
            input=[
                {"role": "system", "content": system_txt},
                {"role": "user", "content": user_txt},
            ],
            max_output_tokens=min(1024, max_output_tokens * 2),
        )
        ans_r = _extract_any(resp_r)
        if ans_r:
            return ans_r
        if debug_logs is not None:
            debug_logs.append("Responses fallback empty")
    except Exception as e:
        if debug_logs is not None:
            debug_logs.append(f"Responses fallback error: {e}")

    print(debug_logs)
    return "（モデルが回答テキストを返しませんでした。デプロイ設定/モデル種類を確認してください）"


def init_session_state():
    ss = st.session_state
    ss.setdefault("pdf_hash", None)
    ss.setdefault("pages_meta", [])
    ss.setdefault("page_count", 0)
    ss.setdefault("current_page", 0)
    ss.setdefault("summary_cache", {})
    ss.setdefault("audio_cache", {})
    ss.setdefault("answer_cache", {})
    ss.setdefault("answer_audio_cache", {})
    ss.setdefault("ui_deploy_summary", DEPLOY_SUMMARY)
    ss.setdefault("ui_deploy_tts", DEPLOY_TTS)
    ss.setdefault("ui_deploy_qa", DEPLOY_QA)
    ss.setdefault("summary_audio_played_pages", set())
    ss.setdefault("answer_audio_autoplay_key", None)
    ss.setdefault("full_pdf_text", "")
    ss.setdefault("full_pdf_chunks", [])
    ss.setdefault("qa_debug_logs", [])
    ss.setdefault("qa_debug_enabled", False)
    # 追加（app2.py と合わせる）
    ss.setdefault("company_site_text", "")
    ss.setdefault("company_site_chunks", [])


init_session_state()
# 固定値の定義
scale = 2.0  # app copy 4.py と同じレンダ倍率
voice = TTS_VOICE  # 以降もこの値を使用
audio_fmt = DEFAULT_AUDIO_FORMAT
cps = TTS_SPEED
minsec = 3.0
# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
audio_bytes = b""  # ← ここで初期化

uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type=["pdf"])

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    new_hash = file_bytes_hash(pdf_bytes)

    # Detect a new PDF and reset states
    if st.session_state.pdf_hash != new_hash:
        st.session_state.pdf_hash = new_hash
        st.session_state.pages_meta = []
        st.session_state.summary_cache.clear()
        st.session_state.audio_cache.clear()
        st.session_state.current_page = 0
        st.session_state.auto_run = False
        st.session_state.auto_started_at = None
        st.session_state.prev_page_for_timer = -1

        # Parse PDF pages (store image + text for each page)
        with st.spinner("PDFを解析しています…"):
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            except Exception as e:
                st.error(f"PDF を開けませんでした: {e}")
                st.stop()

            pages = []
            for p in doc:
                try:
                    page_png = render_pdf_page_pixmap(p, scale=scale)
                    page_b64 = encode_b64_png(page_png)
                except Exception:
                    page_png = b""
                    page_b64 = ""
                page_text = extract_page_text(p)
                pages.append(
                    {
                        "image_png": page_png,
                        "image_b64": page_b64,
                        "text": page_text,
                    }
                )
            st.session_state.pages_meta = pages
            st.session_state.page_count = len(pages)

    page_count = st.session_state.page_count
    if page_count == 0:
        st.error("ページがありません（暗号化PDFなどの可能性）。")
        st.stop()

    # PDF解析後（pages_meta 設定直後）全文とチャンクを生成
    text_all = "\n".join(p.get("text", "") for p in st.session_state.pages_meta)
    st.session_state.full_pdf_text = text_all
    st.session_state.full_pdf_chunks = split_chunks(text_all, size=900, overlap=120)

    idx = st.session_state.current_page
    image_png = st.session_state.pages_meta[idx]["image_png"]
    image_b64 = st.session_state.pages_meta[idx]["image_b64"]
    page_text = st.session_state.pages_meta[idx]["text"]

    # 横並びレイアウト (app copy 4 と同様)
    col_left, col_right = st.columns([7, 3])
    with col_left:
        if image_png:
            st.image(image_png, caption=f"ページ {idx+1}")
        else:
            st.info("このページのプレビュー画像は生成できませんでした。")
        btn_prev, btn_next = st.columns([1, 1])

        def go_prev():
            st.session_state.current_page = max(0, st.session_state.current_page - 1)
            st.session_state.auto_started_at = None

        def go_next():
            st.session_state.current_page = min(page_count - 1, st.session_state.current_page + 1)
            st.session_state.auto_started_at = None

        with btn_prev:
            st.button("⬅️ 前へ", on_click=go_prev, disabled=(st.session_state.current_page == 0), key=f"prev_btn_{idx}")
        with btn_next:
            st.button("次へ ➡️", on_click=go_next, disabled=(st.session_state.current_page >= page_count - 1), key=f"next_btn_{idx}")

    with col_right:
        st.subheader("スライド説明")
        client = get_azure_client()
        if idx not in st.session_state.summary_cache:
            with st.spinner("要約を生成中…"):
                try:
                    # build_multimodal_summary 呼び出し部（temperature 引数削除）
                    summary = build_multimodal_summary(
                        client=client,
                        deployment_name=st.session_state.ui_deploy_summary,
                        page_text=page_text,
                        page_image_b64=image_b64 if image_b64 else None,
                        max_output_tokens=2000,
                        language=st.session_state.ui_lang,
                    )
                    if not summary or not summary.strip():
                        summary = "（要約が空でした。モデル・デプロイ設定をご確認ください）"
                except Exception as e:
                    summary = f"（要約生成でエラーが発生しました：{e}）"
                st.session_state.summary_cache[idx] = summary
        summary = st.session_state.summary_cache.get(idx, "")
        st.write(summary)

        # ================= Q&A =================
        st.subheader("Q&A")
        qa_question_key = f"qa_question_{idx}"
        qa_question = st.text_input("質問を入力してください", key=qa_question_key)

        q_key = (idx, qa_question.strip()) if qa_question.strip() else None

        # ボタン: 回答生成 / クリア （音声化ボタン削除）
        qa_btn_cols = st.columns([1, 1])
        btn_gen, btn_clr = qa_btn_cols

        def ensure_answer_and_tts():
            if not q_key:
                return
            if q_key not in st.session_state.answer_cache:
                with st.spinner("回答を生成中…"):
                    ans = build_slide_qa_answer(
                        client=client,
                        deployment_name=st.session_state.ui_deploy_qa,
                        slide_text=page_text,
                        question=qa_question.strip(),
                        image_b64=image_b64 if image_b64 else None,
                        full_pdf_chunks=st.session_state.full_pdf_chunks,
                        site_chunks=st.session_state.company_site_chunks,
                        language=st.session_state.ui_lang,
                    )
                    st.session_state.answer_cache[q_key] = ans
            ans_txt = st.session_state.answer_cache.get(q_key, "")
            if (not ans_txt) or ans_txt.startswith("（回答生成でエラー"):
                return
            if q_key not in st.session_state.answer_audio_cache:
                with st.spinner("回答音声を生成中…"):
                    audio_ans = synthesize_tts(
                        client=client,
                        deployment_name=st.session_state.ui_deploy_tts,
                        text=ans_txt,
                        voice=voice,
                        audio_format=audio_fmt,
                    )
                    st.session_state.answer_audio_cache[q_key] = audio_ans
            st.session_state.answer_audio_autoplay_key = q_key

        with btn_gen:
            st.button("回答生成", key=f"qa_gen_{idx}", disabled=not qa_question.strip(), on_click=ensure_answer_and_tts)

        def clear_answer():
            if q_key:
                st.session_state.answer_cache.pop(q_key, None)
                st.session_state.answer_audio_cache.pop(q_key, None)
                # フラグがこの質問ならクリア
                if st.session_state.answer_audio_autoplay_key == q_key:
                    st.session_state.answer_audio_autoplay_key = None

        with btn_clr:
            st.button("クリア", key=f"qa_clr_{idx}", disabled=not q_key, on_click=clear_answer)

        current_answer = st.session_state.answer_cache.get(q_key, "") if q_key else ""

        if current_answer:
            st.markdown("**回答:**")
            st.write(current_answer)
            ans_audio = st.session_state.answer_audio_cache.get(q_key, b"")
            autoplay_answer = q_key == st.session_state.answer_audio_autoplay_key
            if ans_audio:
                render_inline_audio(ans_audio, audio_fmt, autoplay=autoplay_answer)
            if autoplay_answer:
                # 一度再生後にフラグ解除（再生継続中に再レンダーしても再自動再生しない）
                st.session_state.answer_audio_autoplay_key = None
        # ================= /Q&A =================

        st.subheader("音声")
        if not summary or not summary.strip() or summary.startswith("（要約生成でエラー"):
            st.session_state.audio_cache[idx] = b""
            st.info("要約が空またはエラーのため、このページの音声生成はスキップしました。")
            audio_bytes = b""
        else:
            if idx not in st.session_state.audio_cache:
                with st.spinner("音声を生成中…"):
                    try:
                        audio_bytes = synthesize_tts(
                            client=client,
                            deployment_name=st.session_state.ui_deploy_tts,
                            text=summary,
                            voice=voice,
                            audio_format=audio_fmt,
                        )
                    except Exception as e:
                        audio_bytes = b""
                        st.error(f"TTS生成でエラー: {e}")
                    st.session_state.audio_cache[idx] = audio_bytes
            audio_bytes = st.session_state.audio_cache.get(idx, b"")

        if audio_bytes:
            # ここで初回のみ autoplay
            played_pages = st.session_state.summary_audio_played_pages
            first_play = idx not in played_pages
            if first_play:
                played_pages.add(idx)
            autoplay_attr = "autoplay " if first_play else ""  # 初回のみ自動再生
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            mime = "audio/ogg" if audio_fmt == "ogg" else f"audio/{audio_fmt}"
            html = f"<audio src='data:{mime};base64,{audio_b64}' {autoplay_attr}controls></audio>"
            components.html(html, height=80)
        else:
            st.info("このページの音声が生成されませんでした。")
else:
    pass  # 何も表示しない場合はこのように
