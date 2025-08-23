import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Sys path:", sys.path)

import yaml, json, os
from topicbank.adapters import fetch_html_list, fetch_pdf_lines, load_csv
from topicbank.safety import safe_language, hard_block, explicit_level
from topicbank.normalize import normalize_item
from topicbank.tagging import tag_topics
from topicbank.dedupe import dedupe

def run_ingest(sources_cfg="config/sources.yaml", safety_cfg="config/safety.yaml", out_path="data/curated/topics.jsonl"):
    sources = yaml.safe_load(open(sources_cfg))["sources"]
    safety = yaml.safe_load(open(safety_cfg))
    # Combine list of strings into a single regex pattern for levels
    lvl2 = "|".join(safety["profanity_keywords"]["sexual_level_2"])
    lvl3 = "|".join(safety["profanity_keywords"]["sexual_level_3"])
    # Keep ban_regex as a list of patterns
    ban_patterns = safety["ban_regex"]

    all_items = []
    for s in sources:
        name = s["name"]; cat = s.get("category_hint"); typ = s["type"]
        url = s.get("url") # Use .get() for optional keys
        raws = []
        print(f"Ingesting from {typ} source: {name}")
        try:
            if typ == "html":
                raws = fetch_html_list(url, name)
            elif typ == "pdf":
                raws = fetch_pdf_lines(url, name)
            elif typ == "csv":
                raws = load_csv(s["path"], text_col="text", source_name=name)
            else:
                print(f"  -> Skipping unknown source type: {typ}")
                continue
        except Exception as e:
            print(f"  -> Failed to fetch or parse {url}. Reason: {e}")
            continue

        print(f"  -> Found {len(raws)} raw items.")

        # safety + normalize + tags
        processed_count = 0
        for r in raws:
            t = r["text"]
            if not safe_language(t, tuple(safety["language"])): continue
            # Pass the list of patterns directly to hard_block
            if hard_block(t, ban_patterns): continue
            expl = explicit_level(t, lvl2, lvl3)
            if not s.get("explicit_allowed", False) and expl >= 2:
                continue

            # Also tag the ingested item itself
            tags = tag_topics(t)
            item = normalize_item(r, category_hint=cat, expl=expl, tags=tags)
            all_items.append(item)
            processed_count += 1
        print(f"  -> Processed {processed_count} valid items.")

    # dedupe
    print(f"Deduplicating {len(all_items)} total items...")
    clean = dedupe(all_items)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in clean:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Curated: {len(clean)} items -> {out_path}")

if __name__ == "__main__":
    run_ingest()
