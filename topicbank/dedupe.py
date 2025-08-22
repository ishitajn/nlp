from simhash import Simhash

def dedupe(items, threshold=3):
    seen = {}
    out = []
    for it in items:
        s = Simhash(it["text"])
        dup = False
        for k,v in seen.items():
            if s.distance(v) <= threshold:
                dup = True; break
        if not dup:
            seen[it["id"]] = s
            out.append(it)
    return out
