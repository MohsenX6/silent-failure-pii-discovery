"""
saudi_entities.py
Format- and checksum-valid generators for Saudi-context PII entities, plus
curated Arabic name lists and perturbation helpers.

All values are SYNTHETIC. No real personal data is used or reproduced.
"""
import random
import string

# ---------------------------------------------------------------------------
# Curated Arabic given / family names (script + transliteration variants).
# A compact but non-trivial curated list (not a handful of repeats).
# Each entry: (arabic_script, [latin transliteration variants])
# ---------------------------------------------------------------------------
GIVEN_NAMES_M = [
    ("محمد", ["Mohammed", "Muhammad", "Mohamed", "Mohammad"]),
    ("أحمد", ["Ahmed", "Ahmad"]),
    ("عبدالله", ["Abdullah", "Abdallah", "Abdulla"]),
    ("عبدالعزيز", ["Abdulaziz", "Abdelaziz", "Abdul Aziz"]),
    ("عبدالرحمن", ["Abdulrahman", "Abdurrahman", "Abdul Rahman"]),
    ("خالد", ["Khalid", "Khaled"]),
    ("سعد", ["Saad", "Sa'ad"]),
    ("فهد", ["Fahad", "Fahd"]),
    ("سلطان", ["Sultan", "Soltan"]),
    ("فيصل", ["Faisal", "Faysal"]),
    ("ناصر", ["Nasser", "Naser", "Nassir"]),
    ("بندر", ["Bandar", "Bander"]),
    ("تركي", ["Turki", "Turky"]),
    ("ماجد", ["Majed", "Majid"]),
    ("يوسف", ["Yousef", "Yusuf", "Youssef"]),
    ("إبراهيم", ["Ibrahim", "Ibraheem"]),
    ("عمر", ["Omar", "Umar"]),
    ("سعود", ["Saud", "Saood"]),
    ("راكان", ["Rakan", "Rakaan"]),
    ("ريان", ["Rayan", "Rayyan"]),
]
GIVEN_NAMES_F = [
    ("نورة", ["Noura", "Nora", "Nourah"]),
    ("سارة", ["Sara", "Sarah"]),
    ("ريم", ["Reem", "Rim"]),
    ("لمى", ["Lama", "Lamah"]),
    ("هند", ["Hind", "Hend"]),
    ("منيرة", ["Munira", "Muneera"]),
    ("الجوهرة", ["Aljawhara", "Al Jawhara"]),
    ("العنود", ["Alanoud", "Al Anoud"]),
    ("شهد", ["Shahad", "Shhd"]),
    ("جنى", ["Jana", "Janah"]),
    ("دانة", ["Dana", "Danah"]),
    ("غادة", ["Ghada", "Ghadah"]),
    ("وفاء", ["Wafa", "Wafaa"]),
    ("أمل", ["Amal", "Amel"]),
    ("رزان", ["Razan", "Razaan"]),
]
FAMILY_NAMES = [
    ("القحطاني", ["Alqahtani", "Al-Qahtani", "Al Qahtani"]),
    ("الغامدي", ["Alghamdi", "Al-Ghamdi", "Al Ghamdi"]),
    ("الشهري", ["Alshehri", "Al-Shehri", "Al Shahri"]),
    ("الدوسري", ["Aldosari", "Al-Dosari", "Al Dossary"]),
    ("العتيبي", ["Alotaibi", "Al-Otaibi", "Al Otaibi"]),
    ("الحربي", ["Alharbi", "Al-Harbi", "Al Harby"]),
    ("المطيري", ["Almutairi", "Al-Mutairi", "Al Mutairi"]),
    ("الزهراني", ["Alzahrani", "Al-Zahrani", "Al Zahrani"]),
    ("السبيعي", ["Alsubaie", "Al-Subaie", "Al Subai'i"]),
    ("الشمري", ["Alshammari", "Al-Shammari", "Al Shamri"]),
    ("البقمي", ["Albaqami", "Al-Baqami"]),
    ("الرشيد", ["Alrashid", "Al-Rashid", "Al Rasheed"]),
    ("آل سعود", ["Al Saud", "Al-Saud"]),
    ("الخالدي", ["Alkhalidi", "Al-Khalidi"]),
    ("العنزي", ["Alanazi", "Al-Anazi", "Al Enezi"]),
    ("الجهني", ["Aljuhani", "Al-Juhani"]),
    ("الفيفي", ["Alfaifi", "Al-Faifi"]),
    ("اليامي", ["Alyami", "Al-Yami"]),
]

WESTERN_GIVEN = ["John", "Michael", "David", "James", "Robert", "Sarah",
                 "Emily", "Jessica", "Daniel", "Thomas", "Laura", "Anna"]
WESTERN_FAMILY = ["Smith", "Johnson", "Brown", "Taylor", "Wilson", "Miller",
                  "Davis", "Clark", "Walker", "Hall", "Young", "King"]

ARABIC_INDIC = {"0": "٠", "1": "١", "2": "٢", "3": "٣", "4": "٤",
                "5": "٥", "6": "٦", "7": "٧", "8": "٨", "9": "٩"}
# Latin -> visually-confusable Cyrillic homoglyphs
HOMOGLYPH = {"A": "А", "a": "а", "e": "е", "o": "о", "O": "О",
             "c": "с", "p": "р", "x": "х", "y": "у", "M": "М",
             "H": "Н", "K": "К", "T": "Т", "B": "В"}
ZWSP = "​"  # zero-width space


# ---------------------------------------------------------------------------
# Checksum-valid identifier generators
# ---------------------------------------------------------------------------
def _luhn_check_digit(num9: str) -> str:
    """Saudi National ID uses the Luhn algorithm over the 10 digits."""
    digits = [int(d) for d in num9]
    total = 0
    # positions (from left, 9 digits) doubled at even index when building 10-digit Luhn
    # Build so that final 10-digit number passes standard Luhn (mod 10 == 0).
    for i, d in enumerate(num9):
        n = int(d)
        # double digits in odd positions counting from rightmost of the final number;
        # with check digit appended, the 9 source digits sit at positions 2..10 from right.
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    check = (10 - (total % 10)) % 10
    return str(check)


def gen_national_id(rng: random.Random, resident: bool = False) -> str:
    lead = "2" if resident else "1"
    body = "".join(str(rng.randint(0, 9)) for _ in range(8))
    num9 = lead + body
    return num9 + _luhn_check_digit(num9)


def validate_national_id(nid: str) -> bool:
    if len(nid) != 10 or not nid.isdigit():
        return False
    return _luhn_check_digit(nid[:9]) == nid[9]


def _mod97(s: str) -> int:
    return int(s) % 97


def gen_iban(rng: random.Random) -> str:
    bank = "".join(str(rng.randint(0, 9)) for _ in range(2))
    account = "".join(str(rng.randint(0, 9)) for _ in range(18))
    bban = bank + account  # 20 digits
    # mod-97: rearrange SA00 + bban, convert letters S=28 A=10
    rearranged = bban + "2810" + "00"
    check = 98 - _mod97(rearranged)
    return f"SA{check:02d}{bban}"


def validate_iban(iban: str) -> bool:
    if not iban.startswith("SA") or len(iban) != 24:
        return False
    rearranged = iban[4:] + "2810" + iban[2:4]
    return _mod97(rearranged) == 1


def gen_phone(rng: random.Random, intl: bool = True) -> str:
    rest = "".join(str(rng.randint(0, 9)) for _ in range(7))
    second = rng.choice("012345689")
    if intl:
        return f"+9665{second}{rest}"
    return f"05{second}{rest}"


def gen_email(rng: random.Random, given_latin: str, family_latin: str) -> str:
    g = given_latin.lower().replace(" ", "").replace("'", "")
    f = family_latin.lower().replace(" ", "").replace("-", "").replace("'", "")
    sep = rng.choice([".", "_", ""])
    dom = rng.choice(["gmail.com", "outlook.sa", "company.com.sa",
                      "ksu.edu.sa", "hotmail.com", "stc.com.sa"])
    num = rng.choice(["", str(rng.randint(1, 99))])
    return f"{g}{sep}{f}{num}@{dom}"


def gen_national_address(rng: random.Random) -> str:
    """Saudi National Address short code: 4 letters + 4 digits."""
    letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(4))
    digits = "".join(str(rng.randint(0, 9)) for _ in range(4))
    return letters + digits


# ---------------------------------------------------------------------------
# Name builders by language stratum
# ---------------------------------------------------------------------------
def pick_name(rng: random.Random, lang: str):
    """Return (display_value, lang_tag). lang in {'arabic','latin','mixed'}."""
    pool_g = GIVEN_NAMES_M if rng.random() < 0.6 else GIVEN_NAMES_F
    g_ar, g_lat = rng.choice(pool_g)
    f_ar, f_lat = rng.choice(FAMILY_NAMES)
    g_lat_v = rng.choice(g_lat)
    f_lat_v = rng.choice(f_lat)
    if lang == "arabic":
        return f"{g_ar} {f_ar}", "arabic", g_lat_v, f_lat_v
    if lang == "latin":
        return f"{g_lat_v} {f_lat_v}", "latin", g_lat_v, f_lat_v
    # mixed / code-switched: Arabic given + Latin family or vice versa
    if rng.random() < 0.5:
        return f"{g_ar} {f_lat_v}", "mixed", g_lat_v, f_lat_v
    return f"{g_lat_v} {f_ar}", "mixed", g_lat_v, f_lat_v


# ---------------------------------------------------------------------------
# Perturbation helpers (return perturbed string)
# ---------------------------------------------------------------------------
def perturb_translit(rng: random.Random, name_latin_g, name_latin_f):
    # alternate transliteration spelling of a Latin name
    return f"{name_latin_g} {name_latin_f}"


def to_arabic_indic(s: str) -> str:
    return "".join(ARABIC_INDIC.get(c, c) for c in s)


def apply_homoglyph(rng: random.Random, s: str, n: int = 2) -> str:
    s = list(s)
    idxs = [i for i, c in enumerate(s) if c in HOMOGLYPH]
    rng.shuffle(idxs)
    for i in idxs[:n]:
        s[i] = HOMOGLYPH[s[i]]
    return "".join(s)


def format_perturb_id(rng: random.Random, nid: str) -> str:
    # insert spaces/dashes
    style = rng.choice(["space", "dash", "zwsp"])
    if style == "space":
        return f"{nid[:4]} {nid[4:7]} {nid[7:]}"
    if style == "dash":
        return f"{nid[:1]}-{nid[1:5]}-{nid[5:]}"
    return nid[:5] + ZWSP + nid[5:]


def format_perturb_iban(rng: random.Random, iban: str) -> str:
    return " ".join(iban[i:i + 4] for i in range(0, len(iban), 4))
