"""
detectors.py
Common interface: each detector takes raw text and returns a list of
predicted spans [{start,end,type}].

Executed detectors (all run fully offline in the evaluation environment):
  - regex_baseline : Saudi-aware, ASCII-only pattern floor (no name detection).
  - presidio       : Microsoft Presidio default pattern recognizers
                     (EMAIL / PHONE / IBAN). No Saudi NATIONAL_ID recognizer and
                     no spaCy NLP engine could be provisioned offline, so PERSON
                     and Saudi structured IDs are not recognised -> reported as-is.
  - gazetteer_en   : dictionary PERSON recogniser built on an English/Western
                     name list (a plausible default for English-centric tooling).
  - gazetteer_loc  : the same recogniser augmented with a curated Saudi/Arabic
                     name list ("locale-tuned" mitigation attempt).

Deferred (designed but NOT executed offline; see paper Sec. Experimental Setup):
  contextual transformer NER (spaCy en_core_web_trf, CAMeL Arabic) and a hosted
  LLM via API. The harness exposes hooks for them; no numbers are fabricated.
"""
import random as _random
import re

import saudi_entities as S

# ---------------------------------------------------------------------------
# Held-out name split for generalization measurement.
# The locale gazetteer is built ONLY from a deterministic 70% "train" portion of
# the curated name lists. The corpus is left unchanged and still draws names
# from the full lists, so any person-name whose given AND family entries both
# fall in the held-out 30% is genuinely out-of-vocabulary for the dictionary.
# This lets the scorer separate in-dictionary recall (a coverage ceiling that is
# true by construction) from held-out recall (real generalization to unseen
# surface forms). Exact-match dictionaries cannot generalize, so held-out recall
# isolates the gap that motivates a contextual neural recognizer.
# ---------------------------------------------------------------------------
SPLIT_SEED = 20260603
TRAIN_FRAC = 0.70


def _split_list(items, frac, rng):
    idx = list(range(len(items)))
    rng.shuffle(idx)
    k = round(len(items) * frac)
    train_idx = set(idx[:k])
    train = [items[i] for i in range(len(items)) if i in train_idx]
    held = [items[i] for i in range(len(items)) if i not in train_idx]
    return train, held


_rng_split = _random.Random(SPLIT_SEED)
_GM_TR, _GM_HO = _split_list(S.GIVEN_NAMES_M, TRAIN_FRAC, _rng_split)
_GF_TR, _GF_HO = _split_list(S.GIVEN_NAMES_F, TRAIN_FRAC, _rng_split)
_FM_TR, _FM_HO = _split_list(S.FAMILY_NAMES, TRAIN_FRAC, _rng_split)
TRAIN_NAME_ENTRIES = _GM_TR + _GF_TR + _FM_TR

# ---------------------------------------------------------------------------
# 1) Regex / rule baseline  (ASCII-only, as a developer would hand-write)
# ---------------------------------------------------------------------------
RE_NID = re.compile(r"(?<![0-9])[12][0-9]{9}(?![0-9])")
RE_IBAN = re.compile(r"\bSA[0-9]{22}\b")
RE_PHONE = re.compile(r"(?<![0-9+])(?:\+9665[0-9]{8}|05[0-9]{8})(?![0-9])")
RE_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
RE_NADDR = re.compile(r"\b[A-Z]{4}[0-9]{4}\b")


def regex_baseline(text):
    out = []
    for rx, t in [(RE_NID, "NATIONAL_ID"), (RE_IBAN, "IBAN"),
                  (RE_PHONE, "PHONE"), (RE_EMAIL, "EMAIL"),
                  (RE_NADDR, "NATIONAL_ADDRESS")]:
        for m in rx.finditer(text):
            out.append({"start": m.start(), "end": m.end(), "type": t})
    return out


# ---------------------------------------------------------------------------
# 2) Microsoft Presidio (offline pattern recognizers)
# ---------------------------------------------------------------------------
_presidio_ready = False
_pres = {}
try:
    from presidio_analyzer.predefined_recognizers import (
        EmailRecognizer, PhoneRecognizer, IbanRecognizer)
    _pres = {
        "EMAIL_ADDRESS": ("EMAIL", EmailRecognizer()),
        "PHONE_NUMBER": ("PHONE", PhoneRecognizer()),
        "IBAN_CODE": ("IBAN", IbanRecognizer()),
    }
    _presidio_ready = True
except Exception as e:  # pragma: no cover
    print("Presidio unavailable:", e)


def presidio(text):
    if not _presidio_ready:
        return []
    out = []
    for ent, (mapped, rec) in _pres.items():
        try:
            for r in rec.analyze(text, entities=[ent], nlp_artifacts=None):
                if r.score and r.score > 0:
                    out.append({"start": r.start, "end": r.end, "type": mapped})
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# 3/4) Gazetteer PERSON recognisers
# ---------------------------------------------------------------------------
def _norm(tok):
    return tok.strip().lower().strip(".,|'")


def _build_en_gazetteer():
    g = set()
    for n in S.WESTERN_GIVEN + S.WESTERN_FAMILY:
        g.add(n.lower())
    # a handful of transliterated given names that genuinely appear in
    # international/English name lists
    for n in ["omar", "sara", "sarah", "adam", "noor", "leen", "maya",
              "yousef", "ibrahim", "khalid", "faisal"]:
        g.add(n)
    return g


def _build_locale_gazetteer():
    # Built from the TRAIN split only (see TRAIN_NAME_ENTRIES above), not the
    # full lists, so the corpus retains a held-out vocabulary the dictionary has
    # never seen.
    g = _build_en_gazetteer()
    for ar, lat in TRAIN_NAME_ENTRIES:
        g.add(ar.lower())
        # also add multi-word arabic family like "آل سعود"
        for part in ar.split():
            g.add(part.lower())
        for v in lat:
            g.add(v.lower())
            for part in v.split():
                g.add(part.lower())
    return g


_GAZ_EN = _build_en_gazetteer()
_GAZ_LOC = _build_locale_gazetteer()

# token pattern keeps Arabic letters, Latin letters, hyphen & apostrophe
_TOK = re.compile(r"[^\s|]+")


def _gazetteer_detect(text, gaz):
    out = []
    toks = list(_TOK.finditer(text))
    i = 0
    while i < len(toks):
        if _norm(toks[i].group()) in gaz:
            start = toks[i].start()
            end = toks[i].end()
            j = i + 1
            # extend over following name tokens (in gaz or capitalised/Arabic)
            while j < len(toks):
                tk = toks[j].group()
                nt = _norm(tk)
                is_arabic = any("؀" <= ch <= "ۿ" for ch in tk)
                if nt in gaz or (tk[:1].isupper() and tk.isalpha()) or is_arabic:
                    end = toks[j].end()
                    j += 1
                    # stop after 3 tokens max
                    if j - i >= 3:
                        break
                else:
                    break
            out.append({"start": start, "end": end, "type": "PERSON_NAME"})
            i = j
        else:
            i += 1
    return out


def gazetteer_en(text):
    return _gazetteer_detect(text, _GAZ_EN)


def gazetteer_loc(text):
    return _gazetteer_detect(text, _GAZ_LOC)


DETECTORS = {
    "Regex baseline": regex_baseline,
    "Presidio (offline)": presidio,
    "Gazetteer-NER (EN)": gazetteer_en,
    "Gazetteer-NER (locale)": gazetteer_loc,
}
