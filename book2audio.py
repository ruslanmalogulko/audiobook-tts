"""EPUB (русский) -> аудиокнига .m4b с главами. Полностью локально: Silero TTS + RUAccent.

Примеры:
  uv run book2audio.py book.epub --list                 # показать главы
  uv run book2audio.py book.epub --chapters 3 --limit 3000   # пробник: 3000 символов из главы 3
  uv run book2audio.py book.epub                        # вся книга (возобновляется после остановки)
"""
import argparse
import hashlib
import re
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, ITEM_COVER, ITEM_IMAGE, epub
from num2words import num2words

SAMPLE_RATE = 48000
MAX_CHUNK_CHARS = 700  # Silero падает на длинных кусках (~1000+ символов)
BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre", "dd", "dt", "div", "td", "th"]
PAUSE_SENTENCE = 0.25
PAUSE_PARAGRAPH = 0.6
PAUSE_TITLE = 1.2
PAUSE_CHAPTER_END = 1.5


# ---------- EPUB -> главы ----------

def extract_chapters(book_path: Path, min_chars: int = 300):
    book = epub.read_epub(str(book_path), options={"ignore_ncx": False})
    toc_titles = {}

    def walk(entries):
        for entry in entries:
            if isinstance(entry, tuple):
                section, children = entry
                if getattr(section, "href", None):
                    toc_titles.setdefault(section.href.split("#")[0], section.title)
                walk(children)
            elif getattr(entry, "href", None):
                toc_titles.setdefault(entry.href.split("#")[0], entry.title)

    walk(book.toc)

    chapters, pending = [], []
    for item_id, _ in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "lxml")
        for junk in soup.find_all(["sup", "script", "style", "rt"]):
            junk.decompose()
        for footnote_link in soup.find_all("a", attrs={"epub:type": "noteref"}):
            footnote_link.decompose()
        paragraphs = []
        for block in soup.find_all(BLOCK_TAGS):
            if block.find(BLOCK_TAGS):  # берём только «листовые» блоки, без дублей
                continue
            text = " ".join(block.get_text(" ").split())
            if text:
                paragraphs.append(text)
        if not paragraphs:
            continue
        title = toc_titles.get(item.get_name()) or (paragraphs[0][:80] if len(paragraphs[0]) < 120 else None)
        pending.append((title, paragraphs))
        if sum(len(p) for p in paragraphs) >= min_chars:
            merged_title = next((t for t, _ in pending if t), f"Часть {len(chapters) + 1}")
            merged = [p for _, ps in pending for p in ps]
            merged_title = merged_title.strip()
            if re.fullmatch(r"\d+|[IVXLCDM]+", merged_title):
                merged_title = f"Глава {merged_title}"
            chapters.append({"title": merged_title, "paragraphs": merged})
            pending = []
    if pending and chapters:
        chapters[-1]["paragraphs"].extend(p for _, ps in pending for p in ps)

    cover = None
    for item in book.get_items():
        if item.get_type() == ITEM_COVER or (
            item.get_type() == ITEM_IMAGE and "cover" in item.get_name().lower()
        ):
            cover = (item.get_name(), item.get_content())
            break
    title = (book.get_metadata("DC", "title") or [[book_path.stem]])[0][0]
    author = (book.get_metadata("DC", "creator") or [[""]])[0][0]
    return {"title": title, "author": author, "cover": cover, "chapters": chapters}


# ---------- нормализация текста ----------

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
ABBREVIATIONS = {
    r"\bт\.\s?е\.": "то есть", r"\bт\.\s?д\.": "так далее", r"\bт\.\s?п\.": "тому подобное",
    r"\bт\.\s?к\.": "так как", r"\bи\s?др\.": "и другие", r"\bи\s?пр\.": "и прочее",
    r"\bсм\.": "смотри", r"\bнапр\.": "например", r"\bгг\.": "годы", r"\bг\.": "год",
    r"\bвв\.": "века", r"\bв\.(?=\s*[,.;)]|\s+[а-я])": "век", r"\bстр\.": "страница", r"\bрис\.": "рисунок",
    r"\bтыс\.": "тысяч", r"\bмлн\b\.?": "миллионов", r"\bмлрд\b\.?": "миллиардов",
    r"\bруб\.": "рублей", r"\bдол\.": "долларов", r"%": " процентов", r"№": "номер ",
    r"\bдр\.": "другие", r"\bок\.": "около",
}
# окончание порядкового -> (падеж, род, мн. число)
ORDINAL_SUFFIXES = {
    "й": ("nominative", "m", False), "ый": ("nominative", "m", False), "ий": ("nominative", "m", False),
    "я": ("nominative", "f", False), "ая": ("nominative", "f", False),
    "е": ("nominative", "n", False), "ое": ("nominative", "n", False),
    "го": ("genitive", "m", False), "му": ("dative", "m", False), "м": ("prepositional", "m", False),
    "ю": ("accusative", "f", False), "ой": ("genitive", "f", False),
    "х": ("prepositional", "m", True), "ми": ("instrumental", "m", True),
}
FRACTION_NAMES = {1: "десятых", 2: "сотых", 3: "тысячных"}


def ordinal(number: int, case="nominative", gender="m", plural=False) -> str:
    return num2words(number, lang="ru", to="ordinal", case=case, gender=gender, plural=plural)


def roman_to_int(token: str):
    if not re.fullmatch(r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})", token):
        return None
    total, prev = 0, 0
    for ch in reversed(token):
        value = ROMAN[ch]
        total = total - value if value < prev else total + value
        prev = max(prev, value)
    return total or None


def number_to_words(match: re.Match) -> str:
    raw = re.sub(r"\s", "", match.group(0))
    try:
        if "," in raw or "." in raw:
            whole, frac = re.split(r"[.,]", raw, maxsplit=1)
            frac_name = FRACTION_NAMES.get(len(frac))
            if frac_name:
                frac_words = num2words(int(frac), lang="ru", gender="f")
                return f"{num2words(int(whole), lang='ru', gender='f')} целых {frac_words} {frac_name}"
            return f"{num2words(int(whole), lang='ru')} и {num2words(int(frac), lang='ru')}"
        return num2words(int(raw), lang="ru")
    except (ValueError, OverflowError):
        return raw


def years_to_words(text: str) -> str:
    year = r"(\d{3,4})"
    # в 1990-1995 гг. / в 1990-1995 годах
    text = re.sub(
        rf"\b([вВ])\s+{year}\s*[–—-]\s*{year}\s*(?:гг\.?|годах)",
        lambda m: f"{m.group(1)} {ordinal(int(m.group(2)), 'prepositional')} - "
        f"{ordinal(int(m.group(3)), 'prepositional')} годах",
        text,
    )
    # в 1984 г. / в 1984 году
    text = re.sub(
        rf"\b([вВ])\s+{year}\s*(?:г\.?|году)(?![а-я])",
        lambda m: f"{m.group(1)} {ordinal(int(m.group(2)), 'prepositional')} году",
        text,
    )
    # 1984 г. / 1984 года  (родительный: «летом 1984 года», «с 1984 г.»)
    text = re.sub(
        rf"\b{year}\s*(?:г\.?|года)(?![а-я])",
        lambda m: f"{ordinal(int(m.group(1)), 'genitive')} года",
        text,
    )
    # в XX в. / в XX веке, XIX в. / XIX века
    roman = r"([IVXLCDM]{1,6})"
    # «в.» в конце предложения: точка нужна и как конец предложения
    text = re.sub(rf"\b{roman}\s*в\.(?=\s+[А-ЯЁ]|\s*$)", r"\1 в.. ", text)
    text = re.sub(
        rf"\b([вВ])\s+{roman}\s*(?:в\.|веке)",
        lambda m: f"{m.group(1)} {ordinal(roman_to_int(m.group(2)) or 0, 'prepositional')} веке"
        if roman_to_int(m.group(2)) else m.group(0),
        text,
    )
    text = re.sub(
        rf"\b{roman}\s*(?:в\.|века)",
        lambda m: f"{ordinal(roman_to_int(m.group(1)), 'genitive')} века" if roman_to_int(m.group(1)) else m.group(0),
        text,
    )
    return text


# ---------- латиница -> кириллица (Silero читает только кириллицу) ----------

LETTER_NAMES = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи", "h": "эйч", "i": "ай",
    "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн", "o": "оу", "p": "пи", "q": "кью", "r": "ар",
    "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабл ю", "x": "экс", "y": "уай", "z": "зед",
}
ENGLISH_WORDS = {
    "a": "э", "am": "эм", "the": "зе", "of": "оф", "on": "он", "to": "ту", "in": "ин", "and": "энд", "is": "из",
    "it": "ит", "you": "ю", "yes": "йес", "no": "ноу", "now": "нау", "tell": "тэл", "doing": "дуинг",
    "game": "гейм", "knight": "найт", "fallen": "фоллен", "empire": "эмпайр", "hero": "хиро", "heroes": "хироуз",
    "dungeon": "данжен", "kingdom": "кингдом", "hard": "хард", "fit": "фит", "task": "таск", "tasks": "таскс",
    "development": "девелопмент", "level": "левел", "leadership": "лидершип", "situation": "ситуэйшн",
    "behavior": "бихейвиор", "behaviour": "бихейвиор", "action": "экшн", "result": "резалт",
    "alternative": "олтернатив", "ownership": "оунершип", "senior": "синьор", "concept": "концепт",
    "artist": "артист", "skills": "скилз", "team": "тим", "culture": "калчер", "homo": "хомо",
    "ludens": "люденс", "feedback": "фидбэк", "feedforward": "фидфорвард", "manager": "менеджер",
    "lead": "лид", "teamlead": "тимлид", "soft": "софт", "people": "пипл", "one": "уан", "high": "хай",
    "low": "лоу", "performance": "перформанс", "review": "ревью", "meeting": "митинг", "deadline": "дедлайн",
    # аббревиатуры, которые читаются словом
    "smart": "смарт", "disc": "диск", "asap": "эйсап", "boff": "бофф", "okr": "о кей ар", "nasa": "наса",
}
TRANSLIT_RULES = [
    ("tion", "шн"), ("sion", "жн"), ("ight", "айт"), ("ough", "оу"), ("ing", "инг"), ("sch", "ск"),
    ("sh", "ш"), ("ch", "ч"), ("th", "т"), ("ph", "ф"), ("ck", "к"), ("qu", "кв"), ("wh", "у"),
    ("ee", "и"), ("ea", "и"), ("oo", "у"), ("ou", "ау"), ("ow", "оу"), ("ay", "эй"), ("ai", "эй"),
    ("ey", "ей"), ("oy", "ой"), ("au", "о"), ("aw", "о"), ("er", "ер"), ("ir", "ер"), ("ur", "ер"),
    ("a", "а"), ("b", "б"), ("c", "к"), ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"), ("h", "х"),
    ("i", "и"), ("j", "дж"), ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"),
    ("q", "к"), ("r", "р"), ("s", "с"), ("t", "т"), ("u", "а"), ("v", "в"), ("w", "в"), ("x", "кс"),
    ("y", "и"), ("z", "з"),
]


def transliterate(word: str) -> str:
    lower = word.lower()
    if lower in ENGLISH_WORDS:
        return ENGLISH_WORDS[lower]
    if word.isupper() and len(word) <= 5 or len(word) == 1:  # аббревиатура / одна буква: по буквам
        return " ".join(LETTER_NAMES[ch] for ch in lower)
    if len(lower) > 3 and lower.endswith("e"):
        lower = lower[:-1]  # немая e
    result, i = "", 0
    while i < len(lower):
        for latin, cyrillic in TRANSLIT_RULES:
            if lower.startswith(latin, i):
                result += cyrillic
                i += len(latin)
                break
        else:
            i += 1
    return result


def latin_to_cyrillic(text: str) -> str:
    return re.sub(r"[A-Za-z]+(?:'[a-z]+)?", lambda m: transliterate(m.group(0).replace("'", "")), text)


def normalize(text: str) -> str:
    text = text.replace("­", "").replace(" ", " ").replace(" ", " ")
    text = re.sub(r"\[\d+\]|\{\d+\}", "", text)  # сноски [12]
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = years_to_words(text)
    for pattern, replacement in ABBREVIATIONS.items():
        # сокращение в конце предложения: точка нужна и как конец предложения
        text = re.sub(pattern + r"(?=\s+[А-ЯЁ]|\s*$)", replacement + ".", text)
        text = re.sub(pattern, replacement, text)
    text = re.sub(
        r"\b(\d+)-(" + "|".join(sorted(ORDINAL_SUFFIXES, key=len, reverse=True)) + r")\b",
        lambda m: ordinal(int(m.group(1)), *ORDINAL_SUFFIXES[m.group(2)]),
        text,
    )
    text = re.sub(r"(\d+)\s*[–—-]\s*(\d+)", r"\1 - \2", text)  # диапазоны 10-15
    text = re.sub(r"\d{1,3}(?:[  ]\d{3})+(?!\d)|\d+(?:[.,]\d+)?", number_to_words, text)
    # римские цифры: только после «Часть/Глава/...» или из 2+ символов I/V/X (иначе «I», «D» - это английский)
    text = re.sub(
        r"(?:(?<=Часть )|(?<=часть )|(?<=Глава )|(?<=глава )|(?<=Том )|(?<=том )|(?<=Книга )|(?<=Раздел ))"
        r"[IVXLCDMІ]{1,7}\b|\b[IVX]{2,7}\b",
        lambda m: num2words(n, lang="ru") if (n := roman_to_int(m.group(0).replace("І", "I"))) else m.group(0),
        text,
    )
    text = latin_to_cyrillic(text)
    text = re.sub(r"[—–]", ", ", text)  # тире -> пауза
    text = re.sub(r"[«»„“”\"]", "", text)
    text = re.sub(r"[*_#|<>=~^`]", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\.([,;:])", r"\1", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    text = re.sub(r"^[,\s]+", "", text)
    return " ".join(text.split())


def split_chunks(paragraph: str, limit: int = MAX_CHUNK_CHARS):
    sentences = re.split(r"(?<=[.!?…])\s+", paragraph)
    chunks, current = [], ""
    for sentence in sentences:
        while len(sentence) > limit:  # очень длинное предложение режем по запятым/пробелам
            cut = max(sentence.rfind(", ", 0, limit), sentence.rfind(" ", 0, limit))
            cut = cut if cut > limit // 3 else limit
            piece, sentence = sentence[: cut + 1].strip(), sentence[cut + 1 :].strip()
            if current:
                chunks.append(current)
                current = ""
            chunks.append(piece)
        if current and len(current) + len(sentence) + 1 > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return [c for c in chunks if re.search(r"[а-яА-ЯёЁ]", c)]


# ---------- синтез ----------

class Narrator:
    def __init__(self, model_id: str, speaker: str, cache_dir: Path, use_accents: bool, tts_rate: int = SAMPLE_RATE,
                 rate: str = "100%", comma_pause_ms: int = 0):
        import torch

        self.torch = torch
        self.model, _ = torch.hub.load(
            "snakers4/silero-models", "silero_tts", language="ru", speaker=model_id, trust_repo=True, verbose=False
        )
        self.model_id, self.speaker = model_id, speaker
        self.tts_rate = tts_rate  # Silero умеет 8/24/48 кГц; на 48 вокодер звенит сильнее
        self.cache_tag = model_id if tts_rate == SAMPLE_RATE else f"{model_id}@{tts_rate}"
        # темп и паузы через SSML: Silero сам по себе почти не держит паузу на запятых
        self.rate, self.comma_pause_ms = rate, comma_pause_ms
        if rate != "100%" or comma_pause_ms:
            self.cache_tag += f"|rate={rate}|comma={comma_pause_ms}"
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.accentizer = None
        if use_accents:
            from ruaccent import RUAccent

            self.accentizer = RUAccent()
            self.accentizer.load(omograph_model_size="turbo3.1", use_dictionary=True)

    def prepare(self, paragraph: str):
        text = normalize(paragraph)
        if self.accentizer and text:
            text = self.accentizer.process_all(text)
        return split_chunks(text)

    def speak(self, chunk: str, speaker: str | None = None) -> np.ndarray:
        speaker = speaker or self.speaker
        key = hashlib.sha1(f"{self.cache_tag}|{speaker}|{chunk}".encode()).hexdigest()
        cached = self.cache_dir / f"{key}.npy"
        if cached.exists():
            return np.load(cached)
        try:
            audio = self.synthesize(chunk, speaker)
        except Exception as error:  # noqa: BLE001 - один кривой кусок не должен ронять книгу
            print(f"\n  ! пропущен кусок ({error.__class__.__name__}): {chunk[:80]}", file=sys.stderr)
            audio = np.zeros(0, dtype=np.float32)
        np.save(cached, audio)
        return audio

    def to_ssml(self, chunk: str) -> str:
        text = chunk.replace("&", " и ")
        if self.comma_pause_ms:
            comma, clause, sentence = self.comma_pause_ms, int(self.comma_pause_ms * 1.5), self.comma_pause_ms * 2
            text = re.sub(r",\s+", f', <break time="{comma}ms"/> ', text)
            text = re.sub(r"([;:])\s+", rf'\1 <break time="{clause}ms"/> ', text)
            text = re.sub(r"([.!?…])\s+(?=\S)", rf'\1 <break time="{sentence}ms"/> ', text)
        if self.rate != "100%":
            text = f'<prosody rate="{self.rate}">{text}</prosody>'
        return f"<speak>{text}</speak>"

    def synthesize(self, chunk: str, speaker: str) -> np.ndarray:
        use_ssml = self.rate != "100%" or self.comma_pause_ms
        with self.torch.no_grad():
            if use_ssml:
                audio = self.model.apply_tts(ssml_text=self.to_ssml(chunk), speaker=speaker, sample_rate=self.tts_rate)
            else:
                audio = self.model.apply_tts(text=chunk, speaker=speaker, sample_rate=self.tts_rate)
        audio = audio.numpy().astype(np.float32)
        if self.tts_rate != SAMPLE_RATE:
            from scipy.signal import resample_poly

            audio = resample_poly(audio, SAMPLE_RATE, self.tts_rate).astype(np.float32)
        return audio


F5_REPO = "Misha24-10/F5-TTS_RUSSIAN"
F5_CHECKPOINTS = {
    "v2": "F5TTS_v1_Base_v2/model_last_inference.safetensors",
    "v4": "F5TTS_v1_Base_v4_winter/model_212000.safetensors",
}
F5_REF_TEXT = "Утро выдалось тихим, и я неспешно шёл на работу, обдумывая предстоящий разговор с командой."


class F5Narrator(Narrator):
    """F5-TTS (русский файнтюн). Голос клонируется по образцу; образцы делает Silero,
    либо можно положить свою запись в refs/<голос>.wav + refs/<голос>.txt."""

    def __init__(self, model_id, speaker, cache_dir, use_accents, checkpoint="v2", nfe_step=32):
        super().__init__(model_id, speaker, cache_dir, use_accents)
        from f5_tts.api import F5TTS
        from huggingface_hub import hf_hub_download

        self.cache_tag = f"f5-{checkpoint}-nfe{nfe_step}"
        self.nfe_step = nfe_step
        self.refs_dir = cache_dir.parent / "refs"
        self.refs_dir.mkdir(exist_ok=True)
        self.f5 = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file=hf_hub_download(F5_REPO, F5_CHECKPOINTS[checkpoint]),
            vocab_file=hf_hub_download(F5_REPO, "F5TTS_v1_Base/vocab.txt"),
            device="mps",
        )

    def reference(self, speaker: str):
        audio_path, text_path = self.refs_dir / f"{speaker}.wav", self.refs_dir / f"{speaker}.txt"
        if not audio_path.exists():
            text = self.prepare(F5_REF_TEXT)[0]
            audio = super().synthesize(text, speaker)
            sf.write(audio_path, np.concatenate([silence(0.2), audio, silence(0.3)]), SAMPLE_RATE)
            text_path.write_text(text)
        return str(audio_path), text_path.read_text().strip()

    def synthesize(self, chunk: str, speaker: str) -> np.ndarray:
        from scipy.signal import resample_poly

        ref_audio, ref_text = self.reference(speaker)
        wav, sample_rate, _ = self.f5.infer(
            ref_file=ref_audio, ref_text=ref_text, gen_text=chunk, nfe_step=self.nfe_step,
            remove_silence=False, show_info=lambda *a, **k: None,
        )
        wav = np.asarray(wav, dtype=np.float32)
        return resample_poly(wav, SAMPLE_RATE, sample_rate).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


# реплика заканчивается знаком препинания, затем « — » и ремарка автора (и наоборот);
# тире внутри фразы («а вам — как об стену горох») без знака препинания перед ним не считается
DIALOGUE_SPLIT = re.compile(r"(?<=[,.!?…])\s[—–]\s")


class VoiceSplitter:
    """Делит абзац на куски рассказчика и реплик. Помнит, открыта ли многоабзацная «цитата»."""

    def __init__(self):
        self.quote_open = False

    def split(self, paragraph: str):
        text = paragraph.strip()
        if self.quote_open:
            if "»" in text:
                self.quote_open = False
            return [("dialogue", text)]
        if re.match(r"^[—–]\s", text):
            parts = DIALOGUE_SPLIT.split(text[1:].strip())
            segments = []
            for i, part in enumerate(parts):
                role = "dialogue" if i % 2 == 0 else "narration"
                # «найм, — это командная работа»: тире внутри речи, а не ремарка (в ремарке есть глагол)
                if role == "narration" and not has_past_verb(part.split(".")[0]):
                    role = "dialogue"
                if segments and segments[-1][0] == role:
                    segments[-1] = (role, f"{segments[-1][1]} — {part}")
                else:
                    segments.append((role, part))
            return segments
        if text.startswith("«"):
            closing = text.find("»")
            if closing == -1:  # письмо/речь на несколько абзацев
                self.quote_open = True
                return [("dialogue", text)]
            if closing >= len(text) - 3:  # весь абзац - одна цитата
                return [("dialogue", text)]
        return [("narration", text)]


_morph = None


def has_past_verb(text: str) -> bool:
    remark_gender("")  # инициализирует анализатор
    return any(
        (tag := _morph.parse(word)[0].tag).POS in ("VERB", "GRND") and (tag.tense == "past" or tag.POS == "GRND")
        for word in re.findall(r"[а-яё]+", text.lower())
    )


def remark_gender(remark: str):
    """Пол говорящего по ремарке: первый глагол прошедшего времени или имя. None - не понятно."""
    global _morph
    if _morph is None:
        import pymorphy3

        _morph = pymorphy3.MorphAnalyzer()
    for word in re.findall(r"[А-Яа-яЁё]+", remark):
        tag = _morph.parse(word)[0].tag
        if (tag.POS == "VERB" and tag.tense == "past") or "Name" in tag:
            if tag.gender in ("masc", "femn"):
                return "male" if tag.gender == "masc" else "female"
    return None


def speech_gender(speech: str):
    """Пол по первому лицу в самой реплике: «я рада», «я уже поняла», «я был»."""
    remark_gender("")  # инициализирует анализатор
    for match in re.finditer(r"\b[Яя]\s+(?:[а-яё]+\s+)?([а-яё]+)", speech):
        tag = _morph.parse(match.group(1))[0].tag
        if (tag.POS == "VERB" and tag.tense == "past") or tag.POS == "ADJS":
            if tag.gender in ("masc", "femn"):
                return "male" if tag.gender == "masc" else "female"
    return None


def assign_voices(paragraphs: list, splitter: "VoiceSplitter"):
    """[(role, text)] на абзац, где role: narration / male / female.

    Пол берётся из ремарки в том же абзаце. Реплики без ремарки получают пол
    ближайшей известной реплики той же чётности в диалоге (собеседники чередуются),
    иначе - мужской."""
    segmented = [splitter.split(p) for p in paragraphs]
    genders = []
    for segments in segmented:
        if not any(role == "dialogue" for role, _ in segments):
            genders.append("narration")
            continue
        remarks = " ".join(text for role, text in segments if role == "narration")
        speech = " ".join(text for role, text in segments if role == "dialogue")
        gender = speech_gender(speech) or (remark_gender(remarks) if remarks else None)
        previous = paragraphs[len(genders) - 1] if genders else ""
        if gender is None and genders and genders[-1] == "narration" and previous.rstrip().endswith(":"):
            last_sentence = re.split(r"(?<=[.!?…])\s+", previous.strip())[-1]
            gender = remark_gender(last_sentence)  # «Маргарита улыбнулась:»
        genders.append(gender)

    index = 0
    while index < len(genders):
        if genders[index] == "narration":
            index += 1
            continue
        end = index
        while end < len(genders) and genders[end] != "narration":
            end += 1
        run = range(index, end)
        for i in run:
            if genders[i] is None:
                same_parity = [genders[j] for j in sorted(run, key=lambda j: abs(j - i))
                               if (j - i) % 2 == 0 and genders[j] in ("male", "female")]
                genders[i] = same_parity[0] if same_parity else "male"
        index = end

    return [
        [(gender if role == "dialogue" else "narration", text) for role, text in segments]
        for segments, gender in zip(segmented, genders)
    ]


DISPLAY_LIMIT = 220  # длина одного субтитра в символах оригинального текста


def display_units(text: str, limit: int = DISPLAY_LIMIT):
    """Режет оригинальный текст на фразы для субтитров. Каждая фраза озвучивается отдельно,
    поэтому её время в аудио известно точно."""
    sentences = [s for s in re.split(r"(?<=[.!?…»])\s+(?=[«—–(\"A-ZА-ЯЁ0-9])", text.strip()) if s]
    units = []
    for sentence in sentences:
        while len(sentence) > limit:
            cut = sentence.rfind(", ", 0, limit)
            cut = cut + 1 if cut > limit // 3 else (sentence.rfind(" ", 0, limit) or limit)
            units.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if units and len(units[-1]) < 40 and len(units[-1]) + len(sentence) < limit:
            units[-1] = f"{units[-1]} {sentence}"
        else:
            units.append(sentence)
    return [u for u in units if u]


def render_chapter(narrator: Narrator, chapter: dict, out_wav: Path, limit: int | None,
                   voices: dict | None = None):
    """Озвучивает главу. Возвращает (длительность, субтитры [(начало, конец, текст, роль)])."""
    pieces, cues, spoken_chars, position = [], [], 0, 0.0
    paragraphs = chapter["paragraphs"]
    title_is_first = paragraphs and paragraphs[0].strip() == chapter["title"]
    if not title_is_first:
        paragraphs = [chapter["title"], *paragraphs]
    total = len(paragraphs)
    if voices:
        voiced = [[("narration", paragraphs[0])], *assign_voices(paragraphs[1:], VoiceSplitter())]
    else:
        voiced = [[("narration", p)] for p in paragraphs]

    def add(audio: np.ndarray):
        nonlocal position
        pieces.append(audio)
        position += len(audio) / SAMPLE_RATE

    for index, paragraph in enumerate(paragraphs):
        if limit and spoken_chars >= limit:
            break
        for role, segment in voiced[index]:
            speaker = (voices or {}).get(role)
            for unit in display_units(segment):
                start = position
                for chunk in narrator.prepare(unit):
                    add(narrator.speak(chunk, speaker))
                    spoken_chars += len(chunk)
                if position > start:
                    shown = f"— {unit}" if role in ("male", "female") else unit
                    cues.append((start, position, shown, "title" if index == 0 else role))
                add(silence(PAUSE_SENTENCE))
        add(silence(PAUSE_TITLE if index == 0 else PAUSE_PARAGRAPH))
        print(f"\r  абзац {index + 1}/{total}", end="", flush=True)
    print()
    add(silence(PAUSE_CHAPTER_END))
    audio = np.concatenate(pieces) if pieces else silence(1)
    sf.write(out_wav, audio, SAMPLE_RATE)
    return len(audio) / SAMPLE_RATE, cues


# ---------- субтитры и видео ----------

ROLE_COLORS = {"narration": (242, 242, 242), "male": (140, 200, 255), "female": (255, 160, 200), "title": (255, 216, 128)}
ROLE_LABELS = {"narration": "рассказчик", "male": "мужской голос", "female": "женский голос", "title": ""}
def find_font(env_var: str, candidates: list) -> str:
    """Шрифт с кириллицей: переменная окружения, иначе первый найденный из списка (macOS, Linux)."""
    import os

    for path in [os.environ.get(env_var), *candidates]:
        if path and Path(path).exists():
            return path
    raise SystemExit(f"Не найден шрифт с кириллицей для видео. Укажите путь в {env_var}=/path/to/font.ttf")


TEXT_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/PTSerif.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/TTF/DejaVuSerif.ttf",
    "C:/Windows/Fonts/georgia.ttf",
]
LABEL_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/PTSans.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
VIDEO_W, VIDEO_H = 1280, 720
TEXT_BOX = (560, 110, 1220, 650)  # x0, y0, x1, y1 области текста справа от обложки
TEXT_BOX_NO_COVER = (160, 110, 1120, 650)  # без обложки текст по центру кадра
VIDEO_FPS = 5


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(cues: list, path: Path):
    path.write_text("".join(
        f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n" for i, (a, b, text, _) in enumerate(cues, 1)
    ))


def make_background(cover: tuple | None, work: Path) -> Path:
    """Размытая затемнённая обложка на весь кадр + чёткая обложка слева."""
    background = work / "background.png"
    if cover:
        cover_path = work / ("cover" + Path(cover[0]).suffix)
        cover_path.write_bytes(cover[1])
        graph = (
            f"[0]scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,crop={VIDEO_W}:{VIDEO_H},"
            f"boxblur=25:3,eq=brightness=-0.38[bg];"
            f"[0]scale=-2:540[fg];[bg][fg]overlay=x=(520-w)/2+20:y=(H-h)/2"
        )
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(cover_path), "-filter_complex", graph]
    else:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=0x15171c:s={VIDEO_W}x{VIDEO_H}"]
    subprocess.run([*cmd, "-frames:v", "1", str(background)], check=True)
    return background


def wrap_text(draw, text: str, font, width: int):
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=font) <= width or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    return [*lines, line] if line else lines


def draw_frame(background, chapter_title: str, text: str, role: str, text_box=TEXT_BOX):
    from PIL import ImageDraw, ImageFont

    frame = background.copy()
    draw = ImageDraw.Draw(frame)
    x0, y0, x1, y1 = text_box
    label_font = ImageFont.truetype(find_font("AUDIOBOOK_LABEL_FONT", LABEL_FONT_CANDIDATES), 22)
    text_font_path = find_font("AUDIOBOOK_TEXT_FONT", TEXT_FONT_CANDIDATES)
    draw.text((x0, 50), chapter_title.upper()[:60], font=label_font, fill=(170, 170, 170))
    for size in (38, 34, 30, 27, 24):  # уменьшаем шрифт, пока текст не влезет
        font = ImageFont.truetype(text_font_path, size)
        lines = wrap_text(draw, text, font, x1 - x0)
        line_height = int(size * 1.4)
        if len(lines) * line_height <= y1 - y0 - 40:
            break
    block_height = len(lines) * line_height
    y = y0 + (y1 - y0 - block_height) // 2
    if ROLE_LABELS[role]:
        draw.text((x0, y - 38), ROLE_LABELS[role], font=label_font, fill=ROLE_COLORS[role])
    for line in lines:
        draw.text((x0, y), line, font=font, fill=ROLE_COLORS[role])
        y += line_height
    return frame


def video_encoder_args():
    """Аппаратный H.264 на Mac (VideoToolbox), иначе libx264."""
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "h264_videotoolbox" in encoders:
        return ["-c:v", "h264_videotoolbox", "-b:v", "400k"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-tune", "stillimage"]


def build_video(audio_path: Path, cues: list, spans: list, total: float, cover, out_path: Path, work: Path):
    """По кадру на фразу; фраза держится на экране до начала следующей."""
    import shutil
    from PIL import Image

    frames_dir = work / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    background = Image.open(make_background(cover, work)).convert("RGB")
    concat = []
    for i, (start, _, text, role) in enumerate(cues):
        start = 0.0 if i == 0 else start
        end = cues[i + 1][0] if i + 1 < len(cues) else total
        title = next((t for t, a, b in spans if a <= start < b), spans[-1][0])
        frame_path = frames_dir / f"{i:06d}.jpg"
        text_box = TEXT_BOX if cover else TEXT_BOX_NO_COVER
        draw_frame(background, title, text, role, text_box).save(frame_path, quality=85)
        concat += [f"file '{frame_path.resolve()}'", f"duration {max(end - start, 0.05):.3f}"]
        if i % 200 == 0:
            print(f"\r  кадры {i + 1}/{len(cues)}", end="", flush=True)
    print()
    concat.append(f"file '{(frames_dir / f'{len(cues) - 1:06d}.jpg').resolve()}'")
    concat_path = work / "frames.txt"
    concat_path.write_text("\n".join(concat) + "\n")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-stats", "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-i", str(audio_path), "-map", "0:v", "-map", "1:a", "-r", str(VIDEO_FPS),
        *video_encoder_args(), "-pix_fmt", "yuv420p", "-c:a", "copy",
        "-shortest", "-movflags", "+faststart", str(out_path),
    ], check=True)
    shutil.rmtree(frames_dir, ignore_errors=True)


# ---------- сборка m4b ----------

# «студийная» обработка (победила в слепом тесте): срез гула, меньше «коробки», больше разборчивости
# и воздуха, de-esser, мягкий компрессор, громкость по стандарту аудиокниг
STUDIO_FILTER = (
    "highpass=f=70,equalizer=f=280:t=q:w=1.2:g=-2.5,equalizer=f=3200:t=q:w=1.4:g=2.5,"
    "treble=g=4:f=7000:t=s,deesser=i=0.35:m=0.5:f=0.5,"
    "acompressor=threshold=-22dB:ratio=2.5:attack=8:release=120:makeup=2,loudnorm=I=-18:TP=-1.5:LRA=7"
)


def build_m4b(meta: dict, chapter_files: list, durations: list, titles: list, out_path: Path, work: Path):
    concat_list = work / "concat.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in chapter_files))
    ffmeta = [";FFMETADATA1", f"title={meta['title']}", f"artist={meta['author']}", "genre=Audiobook"]
    start = 0
    for title, duration in zip(titles, durations):
        end = start + int(duration * 1000)
        ffmeta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={end}", f"title={title}"]
        start = end
    meta_file = work / "chapters.txt"
    meta_file.write_text("\n".join(ffmeta) + "\n")

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list),
           "-i", str(meta_file)]
    cover_args = []
    if meta["cover"]:
        cover_path = work / ("cover" + Path(meta["cover"][0]).suffix)
        cover_path.write_bytes(meta["cover"][1])
        cmd += ["-i", str(cover_path)]
        cover_args = ["-map", "2:v", "-c:v", "copy", "-disposition:v", "attached_pic"]
    cmd += ["-map", "0:a", "-map_metadata", "1", "-map_chapters", "1", *cover_args,
            "-af", STUDIO_FILTER, "-ar", str(SAMPLE_RATE),
            "-c:a", "aac", "-b:a", "128k", "-ac", "1", str(out_path)]
    subprocess.run(cmd, check=True)


def parse_range(spec: str, count: int):
    selected = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            selected.update(range(int(a), int(b) + 1))
        else:
            selected.add(int(part))
    return [i for i in sorted(selected) if 1 <= i <= count]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("epub", type=Path)
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--model", default="v5_ru", help="модель Silero")
    parser.add_argument("--rate", default="100%", help="темп речи Silero в процентах, например 90%%")
    parser.add_argument("--comma-pause", type=int, default=0, help="пауза на запятых, мс (на ;: в 1,5 раза, между предложениями в 2)")
    parser.add_argument("--tts-rate", type=int, choices=[24000, 48000], default=48000,
                        help="частота синтеза Silero (результат всё равно 48 кГц)")
    parser.add_argument("--engine", choices=["silero", "f5"], default="silero")
    parser.add_argument("--f5-checkpoint", choices=sorted(F5_CHECKPOINTS), default="v2")
    parser.add_argument("--f5-steps", type=int, default=32, help="шаги F5: меньше - быстрее, хуже (16-32)")
    parser.add_argument("--speaker", default="eugene", help="голос рассказчика: aidar, baya, kseniya, eugene, xenia")
    parser.add_argument("--male-voice", help="голос мужских реплик (например eugene)")
    parser.add_argument("--female-voice", help="голос женских реплик (например kseniya)")
    parser.add_argument("--name", help="имя итогового файла без расширения")
    parser.add_argument("--video", action="store_true", help="ещё и .mp4 с субтитрами (обложка + текст)")
    parser.add_argument("--chapters", help="например 1-5,8 (нумерация из --list)")
    parser.add_argument("--limit", type=int, help="озвучить только N символов из каждой главы (пробник)")
    parser.add_argument("--no-accents", action="store_true", help="без RUAccent")
    parser.add_argument("--list", action="store_true", help="только показать главы")
    args = parser.parse_args()

    meta = extract_chapters(args.epub)
    chapters = meta["chapters"]
    if args.list:
        print(f"{meta['title']} - {meta['author']}")
        for i, ch in enumerate(chapters, 1):
            print(f"{i:3}. {ch['title'][:70]:70} {sum(len(p) for p in ch['paragraphs']):>8} симв.")
        print(f"Всего: {sum(len(p) for c in chapters for p in c['paragraphs']):,} символов")
        return

    indexes = parse_range(args.chapters, len(chapters)) if args.chapters else list(range(1, len(chapters) + 1))
    slug = re.sub(r"[^\w-]+", "_", args.epub.stem)[:60]
    work = args.out / slug
    (work / "chapters").mkdir(parents=True, exist_ok=True)
    if args.engine == "f5":
        narrator = F5Narrator(args.model, args.speaker, work / "cache", not args.no_accents, args.f5_checkpoint,
                               args.f5_steps)
    else:
        narrator = Narrator(args.model, args.speaker, work / "cache", not args.no_accents, args.tts_rate,
                            args.rate, args.comma_pause)

    voices = None
    if args.male_voice or args.female_voice:
        voices = {"male": args.male_voice or args.speaker, "female": args.female_voice or args.speaker}
    files, durations, titles, cues, spans, offset = [], [], [], [], [], 0.0
    for i in indexes:
        chapter = chapters[i - 1]
        print(f"[{i}/{len(chapters)}] {chapter['title'][:70]}")
        wav = work / "chapters" / f"{i:03d}.wav"
        duration, chapter_cues = render_chapter(narrator, chapter, wav, args.limit, voices)
        cues += [(a + offset, b + offset, text, role) for a, b, text, role in chapter_cues]
        spans.append((chapter["title"], offset, offset + duration))
        offset += duration
        durations.append(duration)
        files.append(wav)
        titles.append(chapter["title"])

    suffix = "" if not (args.chapters or args.limit) else "-sample"
    voice_tag = "+".join([args.speaker, *(v for v in (args.male_voice, args.female_voice) if v)])
    out_file = args.out / (f"{args.name}.m4b" if args.name else f"{slug}-{voice_tag}{suffix}.m4b")
    build_m4b(meta, files, durations, titles, out_file, work)
    srt_path = out_file.with_suffix(".srt")
    write_srt(cues, srt_path)
    print(f"\nГотово: {out_file.resolve()}  ({sum(durations) / 3600:.2f} ч)\nСубтитры: {srt_path.resolve()}")
    if args.video:
        video_path = out_file.with_suffix(".mp4")
        print("Собираю видео...")
        build_video(out_file, cues, spans, sum(durations), meta["cover"], video_path, work)
        print(f"Видео: {video_path.resolve()}")


if __name__ == "__main__":
    main()
