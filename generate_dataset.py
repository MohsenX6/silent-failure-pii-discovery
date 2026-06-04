"""
generate_dataset.py
Builds a synthetic, programmatically-labeled Saudi-context PII corpus with exact
character offsets, across two modalities (structured / free-text), three language
strata (arabic / latin / mixed) and a paired clean vs. adversarial design.

Output: corpus.jsonl  (one JSON object per record)
        Each entity: {start,end,type,value,lang,perturbation}

Run:  python3 generate_dataset.py --seed 20260603
All data is SYNTHETIC. No real personal data is used.
"""
import argparse
import json
import random

import saudi_entities as S

ENTITY_TYPES = ["PERSON_NAME", "NATIONAL_ID", "IBAN", "PHONE", "EMAIL",
                "NATIONAL_ADDRESS"]


class Builder:
    """Accumulates labelled text segments and tracks offsets."""
    def __init__(self):
        self.parts = []
        self.ents = []
        self.pos = 0

    def add(self, s, etype=None, lang="latin", pert="none", value=None):
        start = self.pos
        self.parts.append(s)
        self.pos += len(s)
        if etype:
            self.ents.append({
                "start": start, "end": self.pos, "type": etype,
                "value": value if value is not None else s,
                "lang": lang, "perturbation": pert,
            })

    def text(self):
        return "".join(self.parts)


# ---------------------------------------------------------------------------
# Structured records
# ---------------------------------------------------------------------------
def make_structured(rng, rid, lang, pert="none"):
    name, lang_tag, g_lat, f_lat = S.pick_name(rng, lang)
    nid = S.gen_national_id(rng, resident=bool(rng.getrandbits(1)))
    iban = S.gen_iban(rng)
    phone = S.gen_phone(rng, intl=bool(rng.getrandbits(1)))
    email = S.gen_email(rng, g_lat, f_lat)
    naddr = S.gen_national_address(rng)

    name_disp, nid_disp, iban_disp, phone_disp = name, nid, iban, phone
    name_pert = id_pert = iban_pert = phone_pert = "none"
    notes_extra = None
    misplaced_ent = None

    if pert == "translit":
        # alternate transliteration of the (Latin) name
        name_disp = f"{g_lat} {f_lat}"
        name_pert = "translit"
        lang_tag = "latin"
    elif pert == "codeswitch":
        name_disp, lang_tag, _, _ = S.pick_name(rng, "mixed")
        name_pert = "codeswitch"
    elif pert == "homoglyph":
        if lang_tag != "arabic":
            name_disp = S.apply_homoglyph(rng, name)
            name_pert = "homoglyph"
        nid_disp = S.to_arabic_indic(nid); id_pert = "homoglyph"
        phone_disp = S.to_arabic_indic(phone); phone_pert = "homoglyph"
    elif pert == "format":
        nid_disp = S.format_perturb_id(rng, nid); id_pert = "format"
        iban_disp = S.format_perturb_iban(rng, iban); iban_pert = "format"
        phone_disp = (phone if phone.startswith("+")
                      else "+966" + phone[1:]); phone_pert = "format"
    elif pert == "misplaced":
        # national id is dropped into the free-text notes field instead of column
        notes_extra = nid
        misplaced_ent = True

    b = Builder()
    b.add(f"record_id: {rid} | full_name: ")
    b.add(name_disp, "PERSON_NAME", lang_tag, name_pert, value=name_disp)
    b.add(" | national_id: ")
    if not misplaced_ent:
        b.add(nid_disp, "NATIONAL_ID", "arabic" if id_pert == "homoglyph" else "latin",
              id_pert, value=nid_disp)
    else:
        b.add("N/A")
    b.add(" | iban: ")
    b.add(iban_disp, "IBAN", "latin", iban_pert, value=iban_disp)
    b.add(" | phone: ")
    b.add(phone_disp, "PHONE", "arabic" if phone_pert == "homoglyph" else "latin",
          phone_pert, value=phone_disp)
    b.add(" | email: ")
    b.add(email, "EMAIL", "latin", "none", value=email)
    b.add(" | address: ")
    b.add(naddr, "NATIONAL_ADDRESS", "latin", "none", value=naddr)
    b.add(" | notes: ")
    if misplaced_ent:
        note = "Customer asked to update record; old id on file "
        b.add(note)
        b.add(notes_extra, "NATIONAL_ID", "latin", "misplaced", value=notes_extra)
        b.add(".")
    else:
        b.add(rng.choice([
            "Account verified during onboarding.",
            "Follow-up requested next quarter.",
            "Beneficiary details confirmed by branch.",
            "No outstanding tickets.",
        ]))
    return {"record_id": rid, "modality": "structured",
            "text": b.text(), "entities": b.ents}


# ---------------------------------------------------------------------------
# Free-text records
# ---------------------------------------------------------------------------
FREE_TEMPLATES = [
    ("ticket", "Support ticket #{tk}: Customer {NAME} reports login issues. "
               "Verification phone {PHONE}; contact email {EMAIL}."),
    ("log", "[2026-03-11 09:{mm}] auth.service: user {NAME} (id {NID}) "
            "transfer to {IBAN} flagged for manual review."),
    ("onboarding", "Onboarding note: new joiner {NAME} submitted national ID {NID}. "
                   "Reachable at {PHONE}. National address {NADDR}."),
    ("comment", "Comment: please mask {NAME}'s record before sharing; "
                "email {EMAIL} and IBAN {IBAN} are still visible."),
    ("note", "Reminder: {NAME} (national id {NID}) requested a statement "
             "sent to {EMAIL}."),
]


def make_freetext(rng, rid, lang, pert="none"):
    name, lang_tag, g_lat, f_lat = S.pick_name(rng, lang)
    nid = S.gen_national_id(rng, resident=bool(rng.getrandbits(1)))
    iban = S.gen_iban(rng)
    phone = S.gen_phone(rng, intl=bool(rng.getrandbits(1)))
    email = S.gen_email(rng, g_lat, f_lat)
    naddr = S.gen_national_address(rng)

    name_disp, name_pert = name, "none"
    nid_disp, id_pert = nid, "none"
    phone_disp, phone_pert = phone, "none"
    iban_disp, iban_pert = iban, "none"
    if pert == "translit":
        name_disp = f"{g_lat} {f_lat}"; name_pert = "translit"; lang_tag = "latin"
    elif pert == "codeswitch":
        name_disp, lang_tag, _, _ = S.pick_name(rng, "mixed"); name_pert = "codeswitch"
    elif pert == "homoglyph":
        if lang_tag != "arabic":
            name_disp = S.apply_homoglyph(rng, name); name_pert = "homoglyph"
        nid_disp = S.to_arabic_indic(nid); id_pert = "homoglyph"
        phone_disp = S.to_arabic_indic(phone); phone_pert = "homoglyph"
    elif pert == "format":
        nid_disp = S.format_perturb_id(rng, nid); id_pert = "format"
        iban_disp = S.format_perturb_iban(rng, iban); iban_pert = "format"

    kind, tmpl = rng.choice(FREE_TEMPLATES)
    fields = {
        "tk": rng.randint(10000, 99999), "mm": rng.randint(10, 59),
    }
    slot_meta = {
        "NAME": ("PERSON_NAME", name_disp, lang_tag, name_pert),
        "NID": ("NATIONAL_ID", nid_disp, "arabic" if id_pert == "homoglyph" else "latin", id_pert),
        "PHONE": ("PHONE", phone_disp, "arabic" if phone_pert == "homoglyph" else "latin", phone_pert),
        "EMAIL": ("EMAIL", email, "latin", "none"),
        "IBAN": ("IBAN", iban_disp, "latin", iban_pert),
        "NADDR": ("NATIONAL_ADDRESS", naddr, "latin", "none"),
    }

    # Tokenize template into literal text and {SLOT} markers, build with offsets
    b = Builder()
    buf = tmpl
    i = 0
    while i < len(buf):
        if buf[i] == "{":
            j = buf.index("}", i)
            key = buf[i + 1:j]
            if key in slot_meta:
                etype, val, lg, pt = slot_meta[key]
                b.add(val, etype, lg, pt, value=val)
            else:  # plain field like {tk}/{mm}
                b.add(str(fields[key]))
            i = j + 1
        else:
            k = buf.find("{", i)
            if k == -1:
                b.add(buf[i:]); break
            b.add(buf[i:k]); i = k
    return {"record_id": rid, "modality": "freetext",
            "text": b.text(), "entities": b.ents}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260603)
    ap.add_argument("--out", default="corpus.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    records = []
    n = 0
    langs = ["arabic", "latin", "mixed"]

    # ---- CLEAN set ----
    for _ in range(360):  # structured clean
        lang = langs[n % 3]
        records.append(make_structured(rng, f"S{n:04d}", lang, "none")); n += 1
    for _ in range(330):  # freetext clean
        lang = langs[n % 3]
        records.append(make_freetext(rng, f"F{n:04d}", lang, "none")); n += 1

    # ---- ADVERSARIAL set (paired families) ----
    fam_struct = {
        "translit": 90, "codeswitch": 90, "homoglyph": 90, "format": 90,
        "misplaced": 90,
    }
    fam_free = {
        "translit": 70, "codeswitch": 70, "homoglyph": 70, "format": 70,
    }
    for fam, cnt in fam_struct.items():
        for _ in range(cnt):
            lang = langs[n % 3]
            records.append(make_structured(rng, f"A{n:04d}", lang, fam)); n += 1
    for fam, cnt in fam_free.items():
        for _ in range(cnt):
            lang = langs[n % 3]
            records.append(make_freetext(rng, f"B{n:04d}", lang, fam)); n += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # sanity: verify offsets
    bad = 0
    for r in records:
        for e in r["entities"]:
            if r["text"][e["start"]:e["end"]] != e["value"]:
                bad += 1
    print(f"records={len(records)} entities={sum(len(r['entities']) for r in records)} "
          f"offset_mismatches={bad} seed={args.seed}")


if __name__ == "__main__":
    main()
