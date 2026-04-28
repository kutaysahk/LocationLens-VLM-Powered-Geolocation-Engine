# create virtual environment 
# cd path/location-lens
# uvicorn main:app
# ngrok 

import os, json, re, gc, tempfile, math, asyncio
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import httpx
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ── VLM init ─────────────────────────────────────────────────────────────────
print("Initialising Qwen2.5-VL-3B-Instruct (4-bit quantised)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", quantization_config=bnb_config, device_map="auto")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    print("VLM Engine Online.")
except Exception as e:
    print(f"VLM Init failed: {e}")
    model = processor = None

# ── Prompt ───────────────────────────────────────────────────────────────────
# Asking for per-sign confidence (1-3) is the key fix for hallucinations:
# the VLM itself knows when it is guessing vs. clearly reading a sign.
# We use this confidence to weight the intersection scoring so uncertain
# reads don't anchor the result.
SYSTEM_PROMPT = """You are a precise sign-reading assistant.
Respond ONLY with a valid JSON object — no markdown, no backticks, no explanation.

Format:
{
  "signs": [
    {"name": "Exact text on sign", "confidence": 3},
    {"name": "Another sign",       "confidence": 2}
  ],
  "scene": "One sentence describing what kind of place or street this is."
}

Confidence scale:
  3 = clearly readable, high certainty
  2 = partially visible or slightly blurry but likely correct
  1 = guessed or uncertain — include only if it seems like a real business name

Rules:
- List every readable business name, shop sign, restaurant board, or venue name.
- Order by visual prominence (largest / most central first).
- Include Turkish and English text exactly as written on the sign.
- If nothing is readable, return an empty signs list.
- Do NOT include generic words like "EXIT", "OPEN", "WC", street numbers, or prices.
"""

# ── Categories ────────────────────────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "Restaurant":        ["restaurant","restoran","lokanta","kebap","kebab","doner","pide",
                          "lahmacun","izgara","burger","pizza","bistro","steakhouse","ocakbasi"],
    "Cafe":              ["cafe","kahve","kahveci","coffee","nescafe","patisserie","pastane",
                          "firin","bakery","cay","tea house","espresso","latte"],
    "Bar":               ["bar","pub","tavern","meyhane","birahane","bira","cocktail","lounge"],
    "Hotel":             ["hotel","otel","hostel","motel","resort","inn","suite","butik","apart"],
    "Shop / Market":     ["market","supermarket","bakkal","manav","migros","bim","a101",
                          "carrefour","sok","shop","store","magaza","outlet","boutique"],
    "Pharmacy":          ["eczane","pharmacy","ilac","drug store"],
    "Bank / ATM":        ["bank","banka","atm","bankamatik","akbank","ziraat","garanti",
                          "vakifbank","yapi kredi","halkbank","finansbank"],
    "Hospital / Clinic": ["hospital","hastane","klinik","clinic","poliklinik","saglik",
                          "doktor","doctor","tip merkezi","medical"],
    "School":            ["okul","school","universite","university","kolej","college",
                          "ilkokul","ortaokul","lise","kindergarten"],
    "Mosque / Church":   ["cami","camii","mosque","kilise","church","mescit","katedral"],
    "Residential":       ["apartman","apartment","konut","residence","sitesi","villa",
                          "bina","building","daire","rezidans"],
    "Gas Station":       ["petrol","benzin","gas station","yakit","opet","shell","bp",
                          "total","lukoil","petkim","aytemiz"],
    "Museum / Culture":  ["muze","museum","galeri","gallery","kultur","sanat","art","sergi"],
    "Park / Nature":     ["park","bahce","garden","orman","forest","plaj","beach","gol","sahil"],
    "Transport":         ["otogar","otobus","terminal","istasyon","station","metro",
                          "tramvay","vapur","iskele","pier","havalimani","airport"],
}
OSM_TYPE_MAP = {
    "house":"Residential","apartments":"Residential","residential":"Residential",
    "restaurant":"Restaurant","fast_food":"Restaurant","food_court":"Restaurant",
    "cafe":"Cafe","coffee_shop":"Cafe","bar":"Bar","pub":"Bar",
    "hotel":"Hotel","hostel":"Hotel","supermarket":"Shop / Market",
    "convenience":"Shop / Market","shop":"Shop / Market","pharmacy":"Pharmacy",
    "bank":"Bank / ATM","atm":"Bank / ATM","hospital":"Hospital / Clinic",
    "clinic":"Hospital / Clinic","school":"School","university":"School",
    "mosque":"Mosque / Church","church":"Mosque / Church","fuel":"Gas Station",
    "museum":"Museum / Culture","park":"Park / Nature",
    "bus_station":"Transport","train_station":"Transport","aerodrome":"Transport",
}

def categorize_place(osm_type, osm_category, display_name, extracted_text):
    combined = f"{osm_type} {osm_category} {display_name} {extracted_text}".lower()
    combined = combined.replace('ç','c').replace('ğ','g').replace('ı','i')\
                       .replace('ö','o').replace('ş','s').replace('ü','u')
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in combined: return cat
    for key, cat in OSM_TYPE_MAP.items():
        if key in combined: return cat
    return "Place"

# ── VLM inference ─────────────────────────────────────────────────────────────
def run_vlm(image_path: str) -> dict:
    """
    Returns {"signs": [{"name": str, "confidence": int}], "scene": str}
    confidence 1-3 per sign — used to weight intersection scoring.
    """
    if model is None: raise RuntimeError("VLM model not loaded")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": "Read all visible signs and return the JSON."},
        ]},
    ]
    text_prompt  = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text_prompt], images=image_inputs, padding=True,
                       return_tensors="pt",
                       min_pixels=256*28*28, max_pixels=448*28*28
                      ).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    input_len = inputs["input_ids"].shape[1]
    raw = processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0].strip()
    del inputs; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"  VLM raw output: {raw!r:.500}")

    signs, scene = _parse_vlm_output(raw)
    print(f"  VLM parsed signs: {signs}")
    return {"signs": signs, "scene": scene}


def _parse_vlm_output(raw: str) -> tuple[list[dict], str]:
    """
    Multi-strategy parser — tries progressively looser approaches so that
    a single formatting quirk never silently discards correctly-read signs.

    Strategy order:
      1. Strict JSON on the whole string (model was perfectly compliant)
      2. Extract the FIRST {...} block and parse that (ignores trailing prose)
      3. Find "signs" / "names" array with a targeted regex (handles truncation)
      4. Last resort: pull quoted strings from the raw text as confidence-2 signs
    """
    scene = ""
    signs = []

    # ── Normalise: strip markdown fences, "assistant" prefix, BOM ─────────
    text = raw.strip().lstrip("\ufeff")
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    # Remove leading role label the model sometimes prepends
    text = re.sub(r"^(assistant|system|user)\s*[\n:]?\s*", "", text, flags=re.IGNORECASE)
    text = text.strip()

    # ── Strategy 1 & 2: JSON parse ────────────────────────────────────────
    parsed = None

    # Try whole string first
    try:
        parsed = json.loads(text)
    except Exception:
        pass

    # Try first {...} block (non-greedy up to matching brace)
    if parsed is None:
        # Find the outermost JSON object by counting braces
        try:
            start = text.index("{")
            depth, end = 0, -1
            for i, ch in enumerate(text[start:], start):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                parsed = json.loads(text[start:end+1])
        except Exception:
            pass

    # Extract from parsed dict
    if parsed is not None:
        scene = str(parsed.get("scene", "")).strip()

        if "signs" in parsed:
            for s in parsed["signs"]:
                try:
                    name = str(s.get("name","") or s.get("text","")).strip()
                    conf = int(s.get("confidence", s.get("conf", 2)))
                    conf = max(1, min(3, conf))
                    if name and len(name) >= 2:
                        signs.append({"name": name, "confidence": conf})
                except Exception:
                    continue
        elif "names" in parsed:
            for n in parsed["names"]:
                name = str(n).strip()
                if name and len(name) >= 2:
                    signs.append({"name": name, "confidence": 2})

        if signs:
            return signs, scene

    # ── Strategy 3: regex-extract sign names from partial/truncated JSON ──
    # Looks for "name": "Akbank" or "Akbank" in a list context
    name_matches = re.findall(
        r'"(?:name|text)"\s*:\s*"([^"]{2,60})"',
        text, re.IGNORECASE
    )
    conf_matches = re.findall(
        r'"(?:confidence|conf)"\s*:\s*([1-3])',
        text
    )
    if name_matches:
        for i, name in enumerate(name_matches):
            name = name.strip()
            if name:
                conf = int(conf_matches[i]) if i < len(conf_matches) else 2
                signs.append({"name": name, "confidence": conf})

        # Also try to grab scene
        sm = re.search(r'"scene"\s*:\s*"([^"]{5,})"', text)
        if sm:
            scene = sm.group(1).strip()

        if signs:
            print(f"  [parser] Strategy 3 (regex) rescued {len(signs)} signs")
            return signs, scene

    # ── Strategy 4: last resort — quoted strings that look like names ──────
    # Pick any quoted token 2-40 chars that isn't a JSON key we know about,
    # and doesn't look like a sentence (no spaces beyond 3 words).
    SKIP = {"signs","names","scene","name","text","confidence","conf",
            "assistant","user","system","json"}
    candidates = re.findall(r'"([A-ZÇĞIÖŞÜa-zçğışöü][^"]{1,39})"', text)
    for c in candidates:
        c = c.strip()
        words = c.split()
        if (2 <= len(c) <= 40
                and c.lower() not in SKIP
                and len(words) <= 4          # not a sentence
                and not c[0].islower()):     # starts with capital (likely a name)
            signs.append({"name": c, "confidence": 1})

    if signs:
        print(f"  [parser] Strategy 4 (heuristic) rescued {len(signs)} signs")

    return signs, scene

# ── Geo helpers ───────────────────────────────────────────────────────────────
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

CORROBORATION_RADIUS_M = 200

# ── Nominatim fetch ───────────────────────────────────────────────────────────
NOM_HEADERS = {"User-Agent": "LocationLens/6.0"}

async def fetch_candidates(client: httpx.AsyncClient, query: str,
                           lat=None, lon=None) -> list[dict]:
    """
    Fetch up to 10 OSM candidates for a query.
    With GPS: bounded 300m search first, then global fallback.
    Without GPS: global only.
    """
    results = []

    if lat is not None:
        offset  = 0.003
        viewbox = f"{lon-offset},{lat+offset},{lon+offset},{lat-offset}"
        r = await client.get("https://nominatim.openstreetmap.org/search",
                             params={"q": query, "format": "jsonv2", "limit": 10,
                                     "viewbox": viewbox, "bounded": 1, "addressdetails": 1},
                             headers=NOM_HEADERS)
        results = r.json() if r.status_code == 200 else []

    if not results:
        params = {"q": query, "format": "jsonv2", "limit": 10, "addressdetails": 1}
        if lat is not None:
            params["lat"] = lat; params["lon"] = lon
        r = await client.get("https://nominatim.openstreetmap.org/search",
                             params=params, headers=NOM_HEADERS)
        results = r.json() if r.status_code == 200 else []

    return [
        {"lat": float(x["lat"]), "lon": float(x["lon"]),
         "name":         x.get("name") or x.get("display_name","").split(",")[0],
         "display_name": x.get("display_name",""),
         "type":         x.get("type",""),
         "category":     x.get("category",""),
         "address":      x.get("address",{}),
         "_query":       query}
        for x in results if "lat" in x and "lon" in x
    ]

# ── GPS reverse geocode ───────────────────────────────────────────────────────
async def reverse_geocode(client: httpx.AsyncClient, lat: float, lon: float) -> dict | None:
    r = await client.get(
        "https://nominatim.openstreetmap.org/reverse",
        params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 18, "addressdetails": 1},
        headers=NOM_HEADERS,
    )
    data = r.json() if r.status_code == 200 else {}
    if "error" in data or "lat" not in data:
        return None
    return {
        "lat":          float(data.get("lat", lat)),
        "lon":          float(data.get("lon", lon)),
        "name":         data.get("name") or data.get("display_name","").split(",")[0],
        "display_name": data.get("display_name",""),
        "type":         data.get("type",""),
        "category":     data.get("category",""),
        "address":      data.get("address",{}),
    }

# ── Intersection engine ───────────────────────────────────────────────────────
async def find_intersection(client: httpx.AsyncClient,
                            signs: list[dict],   # [{"name": str, "confidence": int}]
                            lat=None, lon=None) -> dict | None:
    """
    Multi-sign intersection algorithm with VLM confidence weighting.

    Key improvements over previous version:
    1. Uses VLM confidence (1-3) to weight each sign's vote.
       A confidence-1 sign contributes max 0.33× vs a confidence-3 sign.
       This means hallucinated/uncertain signs can't anchor a wrong location.

    2. Every sign is tried as anchor (not just the first 3).
       If VLM hallucinated sign #1 but correctly read signs #2 and #3,
       anchoring on #2 or #3 will score higher because more signs corroborate it.

    3. GPS proximity: if EXIF GPS exists, candidates close to the user get
       a large bonus. But GPS never completely overrides — it just strongly biases.
       (Pure GPS override is handled upstream before this function is called.)

    Scoring for each anchor candidate:
      base    = sign's own confidence weight (conf/3)
      + Σ corroborating_sign_weight  for each other sign within 200m
        where corroborating_weight = (conf/3) * (1 / prominence_rank)
      + GPS proximity bonus (0-3 pts)
    """
    if not signs:
        return None

    names      = [s["name"]       for s in signs[:8]]   # cap at 8
    confs      = [s["confidence"] for s in signs[:8]]

    # Parallel Nominatim requests for all signs
    tasks = [fetch_candidates(client, name, lat, lon) for name in names]
    all_candidates: list[list[dict]] = await asyncio.gather(*tasks)

    print(f"  Intersection: candidates per sign: {[len(c) for c in all_candidates]}")
    print(f"  Sign confidences: {list(zip(names, confs))}")

    # Nothing found at all
    if not any(all_candidates):
        return None

    best_score   = -1.0
    best_anchor  = None
    best_evidence: list[dict] = []

    # Try every sign as anchor (not just top-N)
    for anchor_idx, anchor_cands in enumerate(all_candidates):
        if not anchor_cands:
            continue

        anchor_conf   = confs[anchor_idx]
        anchor_weight = anchor_conf / 3.0   # 0.33 – 1.0

        for anchor in anchor_cands:
            # Base score: the anchor sign's own confidence
            score    = anchor_weight
            evidence = [{"name": names[anchor_idx], "confidence": anchor_conf, "role": "anchor"}]

            # GPS proximity bonus
            if lat is not None:
                d = haversine_m(lat, lon, anchor["lat"], anchor["lon"])
                if   d <  50:  score += 3.0
                elif d < 150:  score += 2.0
                elif d < 300:  score += 1.0
                elif d < 600:  score += 0.3

            # Corroboration from every other sign
            for i, cands in enumerate(all_candidates):
                if i == anchor_idx or not cands:
                    continue

                sign_conf       = confs[i]
                prominence_rank = i + 1               # 1-indexed
                # Weight = confidence × inverse prominence
                # (prominent, high-confidence signs count most)
                vote_weight = (sign_conf / 3.0) * (1.0 / prominence_rank)

                nearest_dist = min(
                    haversine_m(anchor["lat"], anchor["lon"], c["lat"], c["lon"])
                    for c in cands
                )

                if nearest_dist <= CORROBORATION_RADIUS_M:
                    score += vote_weight
                    evidence.append({"name": names[i], "confidence": sign_conf,
                                     "role": "corroborate", "dist_m": round(nearest_dist)})
                elif nearest_dist <= CORROBORATION_RADIUS_M * 2.5:
                    score += vote_weight * 0.25   # weak partial credit

            if score > best_score:
                best_score   = score
                best_anchor  = anchor
                best_evidence = evidence

    if best_anchor is None:
        # Last resort: first result of any sign (sorted by confidence desc)
        sorted_pairs = sorted(zip(confs, all_candidates), key=lambda x: -x[0])
        for conf, cands in sorted_pairs:
            if cands:
                sign_name = names[confs.index(conf)]
                return {**cands[0],
                        "_evidence":      [{"name": sign_name, "confidence": conf, "role": "anchor"}],
                        "_score":         conf / 3.0,
                        "_method":        "Single sign (no corroboration)",
                        "_corroborated":  False}
        return None

    corroborated = sum(1 for e in best_evidence if e["role"] == "corroborate") > 0
    n_corr = sum(1 for e in best_evidence if e["role"] == "corroborate")

    print(f"  Best: '{best_anchor['name']}' score={best_score:.2f} evidence={[e['name'] for e in best_evidence]}")

    return {
        **best_anchor,
        "_evidence":     best_evidence,
        "_score":        round(best_score, 2),
        "_corroborated": corroborated,
        "_method":       (f"Intersection · {n_corr + 1} signs agreed"
                          if corroborated
                          else "Single sign match"),
    }

# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.post("/analyze-scene/")
async def analyze_scene(
    image: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lng: Optional[float] = Form(None),
    heading: Optional[float] = Form(None),
):
    has_gps = lat is not None and lng is not None
    print(f"\n--- [SCAN START] gps={'YES' if has_gps else 'NO'} lat={lat} lng={lng} ---")

    image_bytes = await image.read()
    print(f"  Received: {len(image_bytes):,} bytes")

    # ── VLM inference ─────────────────────────────────────────────────────
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(image_bytes)
        vlm_result = {"signs": [], "scene": ""}
        try:
            vlm_result = run_vlm(tmp_path)
            print(f"  VLM signs : {vlm_result['signs']}")
            print(f"  VLM scene : {vlm_result['scene']}")
        except Exception as e:
            import traceback
            print(f"  VLM error : {e}")
            print(traceback.format_exc())
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass

    signs = vlm_result.get("signs", [])
    scene = vlm_result.get("scene", "")
    names = [s["name"] for s in signs]

    debug_lines = [
        {"text": s["name"], "confidence": s["confidence"],
         "score": float(max(100 - i*10, 10)),
         "conf_label": ["", "Low", "Medium", "High"][s["confidence"]]}
        for i, s in enumerate(signs)
    ]
    extracted_summary = " / ".join(names[:5])

    async with httpx.AsyncClient(timeout=25.0) as client:

        # ══════════════════════════════════════════════════════════════════
        # PATH A — EXIF GPS EXISTS
        # GPS is ground truth. Use reverse geocode immediately for the
        # place name/address, then use sign names only for categorisation.
        # Skip the intersection engine entirely — it adds no value when we
        # already know exactly where the camera was.
        # ══════════════════════════════════════════════════════════════════
        if has_gps:
            print("  PATH A: EXIF GPS → reverse geocode")
            anchor = await reverse_geocode(client, lat, lng)
            if anchor:
                evidence = [{"name": n["name"], "confidence": n["confidence"], "role": "category-hint"}
                            for n in signs[:3]]
                return format_result(
                    anchor, extracted_summary, scene,
                    "EXIF GPS (exact)", names, debug_lines,
                    evidence, confidence_score=5.0,
                    location_source="gps"
                )
            # reverse geocode failed (very rare) → fall through to intersection

        # ══════════════════════════════════════════════════════════════════
        # PATH B — NO GPS: intersection engine on sign names
        # ══════════════════════════════════════════════════════════════════
        if not signs:
            return {"status": "error", "message": "No readable signs found.",
                    "scene": scene, "extracted_text": "", "all_tokens": [],
                    "ocr_lines": [], "location_source": "none"}

        print("  PATH B: No GPS → intersection engine")
        anchor = await find_intersection(client, signs, None, None)

        if anchor:
            return format_result(
                anchor, extracted_summary, scene,
                anchor["_method"], names, debug_lines,
                anchor["_evidence"], anchor["_score"],
                location_source="signs"
            )

    return {"status": "error", "message": "No location match found.",
            "scene": scene, "extracted_text": extracted_summary,
            "all_tokens": names[:10], "ocr_lines": debug_lines,
            "location_source": "none"}

# ── Result formatter ──────────────────────────────────────────────────────────
def format_result(anchor, extracted_text, scene, mode,
                  all_tokens, debug_lines, evidence, confidence_score,
                  location_source="signs"):

    name         = anchor.get("name","")
    display_name = anchor.get("display_name","")
    osm_type     = anchor.get("type","")
    osm_category = anchor.get("category","")
    address      = anchor.get("address",{})
    category     = categorize_place(osm_type, osm_category, display_name, extracted_text)

    # Confidence label
    # GPS path always gets "GPS Exact"
    # Sign intersection: score ≥ 3 → High, ≥ 1.5 → Medium, else Low
    if location_source == "gps":
        conf_label = "GPS Exact"
    elif confidence_score >= 3.0:
        conf_label = "High"
    elif confidence_score >= 1.5:
        conf_label = "Medium"
    elif confidence_score > 0:
        conf_label = "Low"
    else:
        conf_label = "Unverified"

    return {
        "status":            "success",
        "extracted_text":    extracted_text,
        "scene":             scene,
        "all_tokens":        all_tokens[:10],
        "ocr_lines":         debug_lines,
        "search_mode":       mode,
        "evidence":          evidence,
        "confidence_score":  confidence_score,
        "confidence_label":  conf_label,
        "location_source":   location_source,   # "gps" | "signs"
        "place_data": {
            "name":              name,
            "formatted_address": display_name,
            "category":          category,
            "type":              osm_type.replace("_"," ").capitalize(),
            "lat":               float(anchor.get("lat", 0)),
            "lng":               float(anchor.get("lon", 0)),
            "city":    address.get("city") or address.get("town") or address.get("village",""),
            "country": address.get("country",""),
        },
    }
