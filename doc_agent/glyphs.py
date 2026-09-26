import numpy as np
import cv2
from sklearn.cluster import AgglomerativeClustering

GLYPH_BASE = 0xF0000
SIZE = 24


def label(ch):
    o = ord(ch)
    if o >= GLYPH_BASE:
        return f"G{o - GLYPH_BASE:03d}"
    if 0xE000 <= o <= 0xF8FF:
        return f"P{o - 0xE000:03d}"
    return ch


def binarize(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot read image {path}")
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if ink.mean() > 127:
        ink = 255 - ink
    return gray, ink


def components(ink):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes = []
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return boxes, lab
    min_area = max(4, np.median(areas) * 0.08)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            boxes.append([x, y, x + w, y + h, [i]])
    return boxes, lab


def merge_stacked(boxes):
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = True
    while merged:
        merged = False
        out = []
        for b in boxes:
            for m in out:
                ov = min(b[2], m[2]) - max(b[0], m[0])
                narrow = min(b[2] - b[0], m[2] - m[0])
                vgap = max(b[1], m[1]) - min(b[3], m[3])
                hmax = max(b[3] - b[1], m[3] - m[1])
                if narrow > 0 and ov / narrow > 0.6 and vgap < hmax * 0.6:
                    m[0], m[1] = min(m[0], b[0]), min(m[1], b[1])
                    m[2], m[3] = max(m[2], b[2]), max(m[3], b[3])
                    m[4] = m[4] + b[4]
                    merged = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def find_lines(boxes):
    if not boxes:
        return []
    heights = np.array([b[3] - b[1] for b in boxes])
    med_h = np.median(heights)
    order = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    lines, cur, cur_y = [], [], None
    for b in order:
        cy = (b[1] + b[3]) / 2
        if cur and abs(cy - cur_y) > med_h * 0.7:
            lines.append(cur)
            cur = []
        cur.append(b)
        cur_y = np.mean([(c[1] + c[3]) / 2 for c in cur])
    if cur:
        lines.append(cur)
    return [sorted(l, key=lambda b: b[0]) for l in lines]


def glyph_vector(lab, box):
    x0, y0, x1, y1, ids = box
    crop = np.isin(lab[y0:y1, x0:x1], ids).astype(np.float32)
    h, w = crop.shape
    side = max(h, w)
    pad = np.zeros((side, side), np.float32)
    pad[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    small = cv2.resize(pad, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3, 3), 0)
    v = small.flatten()
    v = v - v.mean()
    norm = np.linalg.norm(v) or 1.0
    return np.concatenate([v / norm, [np.log((w + 1) / (h + 1)) * 0.3]])


def word_gap_threshold(gaps):
    g = np.array([x for x in gaps if x > 0], dtype=float)
    if len(g) < 5:
        return None
    g_sorted = np.sort(g)
    best, best_t = -1, None
    for t in np.unique(g_sorted)[1:]:
        a, b = g[g < t], g[g >= t]
        if len(a) < 2 or len(b) < 2:
            continue
        score = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if score > best:
            best, best_t = score, t
    if best_t is None:
        return None
    small, big = g[g < best_t].mean(), g[g >= best_t].mean()
    return best_t if big > small * 1.8 else None


def direction_from_margins(lines):
    if len(lines) < 3:
        return None, "fewer than 3 lines, margins not usable"
    lefts = np.array([l[0][0] for l in lines], float)
    rights = np.array([l[-1][2] for l in lines], float)
    sl, sr = lefts.std(), rights.std()
    if max(sl, sr) < 1e-6:
        return None, "both margins aligned (justified text)"
    if sl < sr * 0.5:
        return "left-to-right", f"left margin aligned (spread {sl:.0f}px) and right ragged ({sr:.0f}px)"
    if sr < sl * 0.5:
        return "right-to-left", f"right margin aligned (spread {sr:.0f}px) and left ragged ({sl:.0f}px)"
    return None, f"margins similar (left {sl:.0f}px, right {sr:.0f}px)"


def transcribe(paths, threshold=0.45, out_sheet=None):
    all_lines, vectors, boxes_all, pages = [], [], [], []
    for p in paths:
        gray, ink = binarize(p)
        boxes, lab = components(ink)
        boxes = merge_stacked(boxes)
        lines = find_lines(boxes)
        for line in lines:
            idx = []
            for b in line:
                vectors.append(glyph_vector(lab, b))
                boxes_all.append((p, b[:4]))
                idx.append(len(vectors) - 1)
            all_lines.append((line, idx))
        pages.append((p, lines))
    if len(vectors) < 2:
        return {"text": "", "n_glyph_tokens": len(vectors), "notes": ["no glyphs found"]}
    X = np.vstack(vectors)
    cl = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold * np.sqrt(2),
                                 linkage="average", metric="euclidean").fit(X)
    raw = cl.labels_
    freq = np.bincount(raw)
    remap = {old: new for new, old in enumerate(np.argsort(-freq))}
    ids = np.array([remap[r] for r in raw])

    gaps = []
    for line, idx in all_lines:
        gaps += [line[i + 1][0] - line[i][2] for i in range(len(line) - 1)]
    gap_t = word_gap_threshold(gaps)

    text_lines = []
    for line, idx in all_lines:
        s = chr(GLYPH_BASE + ids[idx[0]])
        for i in range(1, len(idx)):
            gap = line[i][0] - line[i - 1][2]
            if gap_t is not None and gap >= gap_t:
                s += " "
            s += chr(GLYPH_BASE + ids[idx[i]])
        text_lines.append(s)

    direction, why = direction_from_margins([l for _, lines in pages for l in lines])
    if out_sheet:
        sign_sheet(boxes_all, ids, out_sheet)
    return {
        "text": "\n".join(text_lines),
        "n_glyph_tokens": len(ids),
        "n_sign_types": int(ids.max() + 1),
        "n_lines": len(all_lines),
        "word_gap_px": None if gap_t is None else float(gap_t),
        "direction": direction,
        "direction_reason": why,
        "cluster_threshold": threshold,
        "sheet": out_sheet,
    }


def sign_sheet(boxes_all, ids, out_path, per_row=12, cell=48):
    n = int(ids.max() + 1)
    rows = (n + per_row - 1) // per_row
    sheet = np.full((rows * (cell + 16), per_row * cell), 255, np.uint8)
    cache = {}
    for k in range(n):
        where = np.where(ids == k)[0][0]
        p, (x0, y0, x1, y1) = boxes_all[where]
        if p not in cache:
            cache[p] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        crop = cache[p][y0:y1, x0:x1]
        h, w = crop.shape
        sc = (cell - 8) / max(h, w)
        crop = cv2.resize(crop, (max(1, int(w * sc)), max(1, int(h * sc))))
        r, c = divmod(k, per_row)
        oy, ox = r * (cell + 16) + 4, c * cell + 4
        sheet[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        cv2.putText(sheet, f"G{k:03d}", (c * cell + 2, r * (cell + 16) + cell + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, 0, 1)
    cv2.imwrite(out_path, sheet)
    return out_path


def ocr_known(paths, langs="eng"):
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    texts, confs = [], []
    for p in paths:
        data = pytesseract.image_to_data(Image.open(p), lang=langs, output_type=pytesseract.Output.DICT)
        words = [(w, float(c)) for w, c in zip(data["text"], data["conf"]) if w.strip() and float(c) >= 0]
        confs += [c for _, c in words]
        texts.append(pytesseract.image_to_string(Image.open(p), lang=langs))
    return {"text": "\n".join(texts), "mean_conf": float(np.mean(confs)) if confs else 0.0, "n_words": len(confs)}
