"""
PAVSig Orthographic Transcription Converter
============================================
Reads existing PAVSig manifests (train/val/test) with IPA transcriptions,
replaces the text field with correct Polish orthographic transcriptions,
and writes new manifests to the orto/ subdirectory.

Usage:
    python convert_pavsig_ortho.py

Input:
    /lp-dev/amelia/data/pavsig/{train,val,test}.json

Output:
    /lp-dev/amelia/data/pavsig/orto/{train,val,test}.json

If any filename stem is missing from WORD_MAP, the script will:
  - Print a WARNING with the stem and filepath
  - Skip that entry
  - Write a summary at the end
Add missing stems to WORD_MAP and re-run.
"""

import json
import sys
from pathlib import Path

# =============================================================================
# ORTHOGRAPHIC MAPPING
# filename_stem (no extension, no trailing digit) -> Polish orthographic text
#
# Rules applied:
#   - Diacritics restored: noz->nóż, waz->wąż, roza->róża, zaba->żaba, etc.
#   - Logotomes kept as written (they are phoneme sequences, not real words)
#   - Duplicate filenames (cia2, owoce2) strip the trailing digit -> same word
# =============================================================================

WORD_MAP = {

    # ------------------------------------------------------------------
    # WORDS (51 total in PAVSig corpus)
    # ------------------------------------------------------------------

    # Animals
    "bocian":       "bocian",        # stork
    "biegacz":      "biegacz",       # runner (beetle)
    "dzokej":       "dżokej",        # jockey
    "jeze":         "jeże",          # hedgehogs
    "kaczka":       "kaczka",        # duck
    "pies":         "pies",          # dog
    "roza":         "róża",          # rose (also animal context)
    "waz":          "wąż",           # snake
    "zaba":         "żaba",          # frog
    "zyrafa":       "żyrafa",        # giraffe

    # Objects / everyday items
    "bazie":        "bazie",         # pussy willows
    "cebula":       "cebula",        # onion
    "czapka":       "czapka",        # hat
    "dziadek":      "dziadek",       # grandfather
    "dzwonek":      "dzwonek",       # bell / bluebell
    "kalosze":      "kalosze",       # rubber boots
    "koszyk":       "koszyk",        # basket
    "ksiazka":      "książka",       # book
    "kucharz":      "kucharz",       # cook / chef
    "las":          "las",           # forest
    "lekarz":       "lekarz",        # doctor
    "mazaki":       "mazaki",        # markers / felt-tip pens
    "noz":          "nóż",           # knife
    "owoce":        "owoce",         # fruits
    "parasol":      "parasol",       # umbrella
    "rzeka":        "rzeka",         # river
    "salata":       "sałata",        # lettuce
    "samolot":      "samolot",       # airplane
    "szafa":        "szafa",         # wardrobe
    "szalik":       "szalik",        # scarf
    "sznurek":      "sznurek",       # string / cord
    "szufelka":     "szufelka",      # dustpan / small shovel
    "warzywa":      "warzywa",       # vegetables
    "widelec":      "widelec",       # fork
    "zabawki":      "zabawki",       # toys
    "zegar":        "zegar",         # clock
    "kuchnia":      "kuchnia",       # kitchen (if present)
    "straż":        "straż",         # fire brigade (alt form of strazak)
    "strazak":      "strażak",       # firefighter
    "ciastka":      "ciastka",       # cookies
    "ciasto":       "ciasto",        # cake (alt form)
    "rybka":        "rybka",         # little fish
    "ryba":         "ryba",          # fish
    "sanki":        "sanki",         # sled
    "lyzwy":        "łyżwy",         # ice skates
    "siatka":       "siatka",        # net / bag
    "siatkówka":    "siatkówka",     # volleyball (alt form)
    "jeż":          "jeż",           # hedgehog (singular)
    "żaba":         "żaba",          # frog (with diacritics, if stored this way)
    "stawek":       "stawek",        # small pond
    "staw":         "staw",          # pond
    "zaba2":        "żaba",          # duplicate

    # ------------------------------------------------------------------
    # LOGOTOMES (17 total — one- or two-syllable nonsense syllables)
    # These are phoneme sequences used for articulation testing.
    # Kept as written since they have no standard orthography.
    # ------------------------------------------------------------------

    "ca":           "ca",
    "cia":          "cia",
    "cza":          "cza",
    "drza":         "drza",
    "dza":          "dza",
    "dzia":         "dzia",
    "dża":          "dża",
    "sa":           "sa",
    "sia":          "sia",
    "sza":          "sza",
    "za":           "za",
    "zia":          "zia",
    "ża":           "ża",
    "aca":          "aca",
    "acia":         "acia",
    "acza":         "acza",
    "asa":          "asa",
    "asia":         "asia",
    "asza":         "asza",
    "aza":          "aza",
    "azia":         "azia",

    # ------------------------------------------------------------------
    # ADDITIONAL WORDS discovered from actual manifests
    # ------------------------------------------------------------------
    "kasza":        "kasza",         # porridge / groats
    "koza":         "koza",          # goat
    "lodzie":       "łodzie",        # boats
    "lokiec":       "łokieć",        # elbow
    "pajac":        "pajac",         # clown / jester
    "paz":          "pąż",           # bud (flower bud) — alt: "pąk"
    "sadzawka":     "sadzawka",      # pond / pool
    "taca":         "taca",          # tray
    "wpasie":       "w pasie",       # at the waist
    "zarowka":      "żarówka",       # light bulb
    "ziarno":       "ziarno",        # grain / seed

    # Additional logotomes discovered from actual manifests
    "ia":           "ia",
    "iu":           "iu",
    "radza":        "radza",
    "rza":          "rza",

}


def get_stem(filepath: str) -> str:
    """
    Extract the canonical stem from a PAVSig filepath.
    Strips file extension and trailing digits used for duplicate recordings
    (e.g. 'cia2.wav' -> 'cia', 'owoce2.wav' -> 'owoce').
    Does NOT strip digits from stems that are purely alphabetic with a
    meaningful trailing digit (none expected in PAVSig).
    """
    stem = Path(filepath).stem          # e.g. 'cia2'
    # Strip trailing digits only if stem has non-digit characters
    if stem and stem[-1].isdigit() and not stem.isdigit():
        stem = stem.rstrip("0123456789")
    return stem


def convert_manifest(input_path: Path, output_path: Path) -> tuple[int, int, list]:
    """
    Convert one manifest file.
    Returns (n_written, n_skipped, list_of_missing_stems).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    missing = []

    with open(input_path, encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for lineno, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] JSON parse error at line {lineno}: {e}")
                skipped += 1
                continue

            stem = get_stem(item["audio_filepath"])

            if stem not in WORD_MAP:
                print(f"  [WARN] No mapping for stem '{stem}' "
                      f"— {item['audio_filepath']}")
                missing.append(stem)
                skipped += 1
                continue

            # Replace text with orthographic transcription
            item["text"] = WORD_MAP[stem]

            # Write with ensure_ascii=False to preserve Polish characters
            json.dump(item, f_out, ensure_ascii=False)
            f_out.write("\n")
            written += 1

    return written, skipped, missing


def discover_stems(input_paths: list[Path]) -> set[str]:
    """Collect all unique stems from a list of manifest files."""
    stems = set()
    for path in input_paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    stems.add(get_stem(item["audio_filepath"]))
                except Exception:
                    pass
    return stems


# =============================================================================
# MAIN
# =============================================================================

def main():
    base_in  = Path("/lp-dev/amelia/data/pavsig")
    base_out = Path("/lp-dev/amelia/data/pavsig/orto")

    splits = ["train", "val", "test"]

    # -------------------------------------------------------------------------
    # Step 0: Discover all stems across all splits and warn about any not in map
    # -------------------------------------------------------------------------
    all_input_paths = [base_in / f"{split}.json" for split in splits]
    existing_paths  = [p for p in all_input_paths if p.exists()]

    if not existing_paths:
        print(f"ERROR: No manifest files found in {base_in}")
        print(f"       Expected: {[str(p) for p in all_input_paths]}")
        sys.exit(1)

    print("=" * 60)
    print("PAVSig Orthographic Transcription Converter")
    print("=" * 60)

    all_stems = discover_stems(existing_paths)
    unknown   = sorted(all_stems - set(WORD_MAP.keys()))

    if unknown:
        print(f"\n[PRE-CHECK] {len(unknown)} stems not in WORD_MAP:")
        for s in unknown:
            print(f"    '{s}'")
        print("\nAdd these to WORD_MAP and re-run.\n")
    else:
        print(f"\n[PRE-CHECK] All {len(all_stems)} stems have mappings. ✓\n")

    # -------------------------------------------------------------------------
    # Step 1: Convert each split
    # -------------------------------------------------------------------------
    total_written = 0
    total_skipped = 0
    all_missing   = []

    for split in splits:
        input_path  = base_in  / f"{split}.json"
        output_path = base_out / f"{split}.json"

        if not input_path.exists():
            print(f"[SKIP] {input_path} does not exist — skipping.")
            continue

        print(f"Converting {split}...")
        written, skipped, missing = convert_manifest(input_path, output_path)
        all_missing.extend(missing)

        print(f"  -> {output_path}")
        print(f"     Written: {written}  |  Skipped: {skipped}")
        total_written += written
        total_skipped += skipped

    # -------------------------------------------------------------------------
    # Step 2: Summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total written : {total_written}")
    print(f"Total skipped : {total_skipped}")

    if all_missing:
        unique_missing = sorted(set(all_missing))
        print(f"\nMissing stems ({len(unique_missing)}) — add to WORD_MAP:")
        for s in unique_missing:
            print(f'    "{s}": "",  # TODO')
    else:
        print("\nAll entries converted successfully. ✓")
        print(f"\nOutput directory: {base_out}")


if __name__ == "__main__":
    main()