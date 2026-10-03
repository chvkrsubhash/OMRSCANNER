import os, json, sqlite3, hashlib, secrets, re
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, abort
from werkzeug.utils import secure_filename
import fitz
import cv2
import numpy as np
from PIL import Image
import io

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "instance")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "omr.sqlite3")
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "replace-this-with-a-long-random-secret")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT UNIQUE NOT NULL,
          email TEXT UNIQUE NOT NULL,
          password_hash TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS templates (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          name TEXT NOT NULL,
          config_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS results (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          template_id INTEGER REFERENCES templates(id) ON DELETE SET NULL,
          test_name TEXT NOT NULL,
          filename TEXT NOT NULL,
          question_count INTEGER NOT NULL,
          correct_count INTEGER NOT NULL,
          incorrect_count INTEGER NOT NULL,
          unanswered_count INTEGER NOT NULL,
          score REAL NOT NULL,
          details_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        """)

def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000).hex()
    return f"{salt}${digest}"

def verify_password(password, stored):
    try:
        salt, digest = stored.split("$", 1)
        return secrets.compare_digest(hash_password(password, salt).split("$", 1)[1], digest)
    except Exception:
        return False

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user_template(template_id, user_id):
    with db() as con:
        return con.execute("SELECT * FROM templates WHERE id=? AND user_id=?", (template_id, user_id)).fetchone()

def normalize_config(form):
    count = int(form.get("question_count", "50"))
    if not 1 <= count <= 500:
        raise ValueError("Question count must be between 1 and 500.")
    options = [x.strip().upper() for x in form.get("options", "A,B,C,D").split(",") if x.strip()]
    if not 2 <= len(options) <= 8 or len(set(options)) != len(options):
        raise ValueError("Provide 2–8 unique answer options, e.g. A,B,C,D.")
    # Coordinates are normalized to the rendered page: 0.0–1.0, measured from top-left.
    cfg = {
        "question_count": count,
        "options": options,
        "page_number": int(form.get("page_number", "1")),
        "layout_mode": form.get("layout_mode", "grid"),
        "x_start": float(form.get("x_start", "0.20")),
        "y_start": float(form.get("y_start", "0.20")),
        "x_step": float(form.get("x_step", "0.04")),
        "y_step": float(form.get("y_step", "0.025")),
        "questions_per_column": int(form.get("questions_per_column", str(count))),
        "option_step": float(form.get("option_step", "0.025")),
        "bubble_radius": int(form.get("bubble_radius", "9")),
        "darkness_threshold": int(form.get("darkness_threshold", "90")),
        "min_fill_ratio": float(form.get("min_fill_ratio", "0.18")),
        "max_fill_ratio": float(form.get("max_fill_ratio", "0.70")),
    }
    if cfg["page_number"] < 1 or cfg["questions_per_column"] < 1:
        raise ValueError("Page number and questions per column must be positive.")
    for key in ("x_start", "y_start", "x_step", "y_step", "option_step", "min_fill_ratio", "max_fill_ratio"):
        if not 0 <= cfg[key] <= 1:
            raise ValueError(f"{key} must be between 0 and 1.")
    if cfg["bubble_radius"] < 2 or cfg["bubble_radius"] > 80:
        raise ValueError("Bubble radius must be between 2 and 80 pixels.")
    if cfg["darkness_threshold"] < 0 or cfg["darkness_threshold"] > 255:
        raise ValueError("Darkness threshold must be between 0 and 255.")
    if cfg["min_fill_ratio"] >= cfg["max_fill_ratio"]:
        raise ValueError("Minimum fill ratio must be lower than maximum.")
    return cfg

def render_pdf_page(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    if page_number < 1 or page_number > len(doc):
        doc.close()
        raise ValueError(f"Template expects page {page_number}, but the uploaded PDF has {len(doc)} page(s).")
    page = doc[page_number - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    doc.close()
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def read_acroform_answers(pdf_path, question_count, options):
    """Read interactive PDF radio/check fields when their field names contain a question number.
    Supported examples: Q1, question_1, Q1_A, question1_B; selected export values must match an option.
    """
    answers = {}
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            widgets = page.widgets()
            if not widgets:
                continue
            for w in widgets:
                name = (w.field_name or "").strip()
                value = str(w.field_value or "").strip()
                if not name or not value or value.lower() in {"off", "false", "0", "none"}:
                    continue
                m = re.search(r"(?:question|q)[ _-]*(\d+)", name, re.I)
                if not m:
                    continue
                q = int(m.group(1))
                if not 1 <= q <= question_count:
                    continue
                # Some radio groups store the selected option in the field value; others encode it in the name.
                candidates = [value.upper(), name.upper().split("_")[-1]]
                selected = next((c for c in candidates if c in options), None)
                if selected:
                    answers[q] = selected
    finally:
        doc.close()
    return answers

def scan_answers(pdf_path, cfg):
    options = cfg["options"]
    count = cfg["question_count"]
    # Interactive AcroForm fields are more reliable than visual detection if usable fields exist.
    form_answers = read_acroform_answers(pdf_path, count, options)
    if len(form_answers) >= max(1, int(count * 0.5)):
        return form_answers, {q: {"method": "pdf_form", "confidence": 1.0, "ratios": {}} for q in form_answers}, "pdf_form"

    # Multi-page visual scanning support:
    qpc = max(1, cfg.get("questions_per_column", 25))
    max_cols = max(1, cfg.get("max_cols_per_page", 4))
    q_per_page = max_cols * qpc

    doc = fitz.open(pdf_path)
    total_doc_pages = len(doc)
    doc.close()

    base_page = max(1, cfg.get("page_number", 1))
    page_cache = {}

    def get_page_gray(p_num):
        if p_num not in page_cache:
            if p_num <= total_doc_pages:
                page_cache[p_num] = render_pdf_page(pdf_path, p_num)
            else:
                page_cache[p_num] = None
        return page_cache[p_num]

    answers, diagnostics = {}, {}
    for q in range(1, count + 1):
        idx = q - 1
        page_offset = idx // q_per_page
        q_in_page = idx % q_per_page
        col = q_in_page // qpc
        row = q_in_page % qpc

        target_p = base_page + page_offset
        gray = get_page_gray(target_p)
        if gray is None:
            diagnostics[q] = {
                "method": "visual",
                "confidence": 0.0,
                "ratios": {opt: 0.0 for opt in options},
                "status": f"page_{target_p}_missing"
            }
            continue

        h, w = gray.shape
        cx0 = int((cfg["x_start"] + col * cfg["x_step"] * (len(options) + 1)) * w)
        cy = int((cfg["y_start"] + row * cfg["y_step"]) * h)
        ratios = {}
        for oi, option in enumerate(options):
            cx = cx0 + int(oi * cfg["option_step"] * w)
            r = cfg["bubble_radius"]
            x1, x2 = max(0, cx-r), min(w, cx+r+1)
            y1, y2 = max(0, cy-r), min(h, cy+r+1)
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                ratios[option] = 0.0
                continue
            # Exclude outer border by sampling a central disk.
            yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
            mask = (xx - (roi.shape[1]-1)/2)**2 + (yy - (roi.shape[0]-1)/2)**2 <= max(1, (r*0.62)**2)
            vals = roi[mask]
            ratios[option] = float(np.mean(vals < cfg["darkness_threshold"])) if vals.size else 0.0
        ranked = sorted(ratios.items(), key=lambda kv: kv[1], reverse=True)
        top_opt, top_ratio = ranked[0]
        second_ratio = ranked[1][1] if len(ranked) > 1 else 0
        marked = [opt for opt, ratio in ratios.items() if ratio >= cfg["min_fill_ratio"]]
        if top_ratio >= cfg["max_fill_ratio"] or (top_ratio >= cfg["min_fill_ratio"] and top_ratio - second_ratio >= 0.08):
            if len(marked) == 1 or (len(marked) > 1 and top_ratio - second_ratio >= 0.12):
                answers[q] = top_opt
        diagnostics[q] = {
            "method": "visual",
            "confidence": round(max(0.0, min(1.0, top_ratio - second_ratio)), 3),
            "ratios": {k: round(v, 3) for k, v in ratios.items()},
            "status": "detected" if q in answers else ("multiple_or_unclear" if len(marked) > 1 else "blank_or_unclear")
        }
    return answers, diagnostics, "visual"

def generate_omr_pdf(template_name, cfg, interactive=False):
    """Generate a printable or digital fillable A4 OMR sheet PDF matching the exact template bubble coordinates.
    Automatically handles multi-page generation when questions exceed 100 (4 columns x 25 rows) so questions
    never overlap or clip with borders."""
    doc = fitz.open()
    pw, ph = 595.3, 841.9  # Standard A4 dimensions in points
    m = 24.0
    L = m + 18.0
    R = pw - m - 18.0
    CW = R - L

    options = cfg.get("options", ["A", "B", "C", "D"])
    count = cfg.get("question_count", 100)
    qpc = max(1, cfg.get("questions_per_column", 25))
    max_cols = max(1, cfg.get("max_cols_per_page", 4))
    q_per_page = max_cols * qpc
    total_pages = max(1, (count + q_per_page - 1) // q_per_page)

    x_start = cfg.get("x_start", 0.136)
    y_start = cfg.get("y_start", 0.238)
    x_step = cfg.get("x_step", 0.0433)
    y_step = cfg.get("y_step", 0.0248)
    opt_step = cfg.get("option_step", 0.0415)
    bub_r_px = cfg.get("bubble_radius", 8)
    bub_r_pt = bub_r_px / 2.0
    n_opts = len(options)

    # ── COLOR PALETTE (Clean Navy / Slate Exam Theme) ─────────────────────────
    NAVY = (0.12, 0.20, 0.32)
    DKINK = (0.15, 0.18, 0.24)
    GRAY = (0.42, 0.46, 0.52)
    LINE_CLR = (0.72, 0.76, 0.82)
    HDR_FILL = (0.92, 0.95, 0.98)
    HDR_BORDER = (0.78, 0.84, 0.90)
    BUB_CLR = (0.25, 0.35, 0.48)

    clean_title = (template_name or "APPSC WEEKLY TEST").replace("_", " ").strip().upper()

    for page_idx in range(total_pages):
        page = doc.new_page(width=pw, height=ph)

        # 1. Corner fiducial registration markers (solid black squares)
        page.draw_rect(fitz.Rect(m, m, m + 11.0, m + 11.0), color=(0,0,0), fill=(0,0,0))
        page.draw_rect(fitz.Rect(pw - m - 11.0, m, pw - m, m + 11.0), color=(0,0,0), fill=(0,0,0))
        page.draw_rect(fitz.Rect(m, ph - m - 11.0, m, ph - m), color=(0,0,0), fill=(0,0,0))
        page.draw_rect(fitz.Rect(pw - m - 11.0, ph - m - 11.0, pw - m, ph - m), color=(0,0,0), fill=(0,0,0))

        # 2. Left-edge timing track
        for ty in range(int(m + 80), int(ph - m - 30), 18):
            page.draw_rect(fitz.Rect(m, ty, m + 6.0, ty + 5.0), color=(0,0,0), fill=(0,0,0))

        p_start = page_idx * q_per_page + 1
        p_end = min(count, (page_idx + 1) * q_per_page)
        p_count = p_end - p_start + 1
        cols_on_page = (p_count + qpc - 1) // qpc

        # ── 3. HEADER BOX ──────────────────────────────────────────────────────
        h_top = 34.0
        h_bot = 154.0
        page.draw_rect(fitz.Rect(L, h_top, R, h_bot), color=LINE_CLR, width=0.9, fill=(0.985, 0.99, 1.0))
        div_x = L + CW * 0.52
        page.draw_line(fitz.Point(div_x, h_top), fitz.Point(div_x, h_bot), color=LINE_CLR, width=0.8)

        # Left Header sub-box
        page.insert_text(fitz.Point(L + 12, h_top + 22), clean_title, fontsize=13, fontname="hebo", color=NAVY)
        sub_text = "OFFICIAL MULTIPLE CHOICE OMR ANSWER SHEET" if page_idx == 0 else f"OFFICIAL MULTIPLE CHOICE OMR ANSWER SHEET · PAGE {page_idx + 1} OF {total_pages}"
        page.insert_text(fitz.Point(L + 12, h_top + 36), sub_text, fontsize=8, fontname="hebo", color=GRAY)
        page.insert_text(fitz.Point(L + 12, h_top + 54), "• Use Blue / Black Ballpoint Pen only. Darken bubbles completely.", fontsize=7.2, fontname="helv", color=DKINK)
        page.insert_text(fitz.Point(L + 12, h_top + 67), "• Do not fold, tear, or use whiteout. One response per question.", fontsize=7.2, fontname="helv", color=DKINK)

        # Guide: Correct ●  Wrong ⊗
        page.insert_text(fitz.Point(L + 12, h_top + 92), "Guide:  Correct", fontsize=7.5, fontname="hebo", color=DKINK)
        page.draw_circle(fitz.Point(L + 76, h_top + 89), 4.2, color=(0,0,0), fill=(0,0,0))
        page.insert_text(fitz.Point(L + 86, h_top + 92), "Wrong", fontsize=7.5, fontname="hebo", color=DKINK)
        page.draw_circle(fitz.Point(L + 123, h_top + 89), 4.2, color=GRAY, width=0.8)
        page.draw_line(fitz.Point(L + 120, h_top + 86), fitz.Point(L + 126, h_top + 92), color=GRAY, width=0.8)
        page.draw_line(fitz.Point(L + 120, h_top + 92), fitz.Point(L + 126, h_top + 86), color=GRAY, width=0.8)

        # Right Header sub-box (Candidate Info & Roll Number Boxes)
        rx0 = div_x + 12
        rx_end = R - 12
        page.insert_text(fitz.Point(rx0, h_top + 22), "CANDIDATE NAME:", fontsize=7.8, fontname="hebo", color=DKINK)
        page.draw_line(fitz.Point(rx0 + 82, h_top + 23), fitz.Point(rx_end, h_top + 23), color=LINE_CLR, width=0.6)
        if interactive:
            w = fitz.Widget()
            w.field_name = f"CandidateName_P{page_idx + 1}"
            w.field_type = fitz.PDF_WIDGET_TYPE_TEXT
            w.rect = fitz.Rect(rx0 + 82, h_top + 10, rx_end, h_top + 23)
            page.add_widget(w)

        # Roll / Reg No digit boxes [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
        page.insert_text(fitz.Point(rx0, h_top + 46), "ROLL / REG NO:", fontsize=7.8, fontname="hebo", color=DKINK)
        bx_start = rx0 + 82
        for d in range(10):
            bx = bx_start + d * 14.5
            page.draw_rect(fitz.Rect(bx, h_top + 35, bx + 12, h_top + 48), color=LINE_CLR, width=0.8, fill=(1,1,1))
            if interactive:
                w = fitz.Widget()
                w.field_name = f"RollDigit_P{page_idx + 1}_{d + 1}"
                w.field_type = fitz.PDF_WIDGET_TYPE_TEXT
                w.rect = fitz.Rect(bx, h_top + 35, bx + 12, h_top + 48)
                page.add_widget(w)

        page.insert_text(fitz.Point(rx0, h_top + 72), "DATE:", fontsize=7.8, fontname="hebo", color=DKINK)
        page.draw_line(fitz.Point(rx0 + 32, h_top + 73), fitz.Point(rx0 + 110, h_top + 73), color=LINE_CLR, width=0.6)
        page.insert_text(fitz.Point(rx0 + 118, h_top + 72), "BATCH / SET:", fontsize=7.8, fontname="hebo", color=DKINK)
        page.draw_line(fitz.Point(rx0 + 180, h_top + 73), fitz.Point(rx_end, h_top + 73), color=LINE_CLR, width=0.6)

        page.insert_text(fitz.Point(rx0, h_top + 98), "CANDIDATE SIGNATURE:", fontsize=7.8, fontname="hebo", color=DKINK)
        page.draw_line(fitz.Point(rx0 + 105, h_top + 99), fitz.Point(rx_end, h_top + 99), color=LINE_CLR, width=0.6)

        # ── 4. COLUMN HEADERS (Light blue-gray pill bar) ────────────────────────
        hdr_y1 = y_start * ph - 22.0
        hdr_y2 = y_start * ph - 6.0
        for col in range(cols_on_page):
            cx0 = (x_start + col * x_step * (n_opts + 1)) * pw
            last_cx = cx0 + (n_opts - 1) * opt_step * pw
            page.draw_rect(fitz.Rect(cx0 - bub_r_pt - 18, hdr_y1, last_cx + bub_r_pt + 6, hdr_y2),
                           color=HDR_BORDER, fill=HDR_FILL, width=0.7)
            page.insert_text(fitz.Point(cx0 - bub_r_pt - 14, hdr_y1 + 11), "Q#", fontsize=7.2, fontname="hebo", color=NAVY)
            for oi, opt in enumerate(options):
                cx = cx0 + oi * opt_step * pw
                page.insert_text(fitz.Point(cx - 2.8, hdr_y1 + 11), opt, fontsize=7.2, fontname="hebo", color=NAVY)

        # ── 5. QUESTION ROWS ───────────────────────────────────────────────────
        for q in range(p_start, p_end + 1):
            local_idx = q - p_start
            col = local_idx // qpc
            row = local_idx % qpc
            cx0 = (x_start + col * x_step * (n_opts + 1)) * pw
            cy = (y_start + row * y_step) * ph

            page.insert_text(fitz.Point(cx0 - bub_r_pt - 16, cy + 2.5), f"{q:02d}", fontsize=6.8, fontname="helv", color=DKINK)
            for oi, opt in enumerate(options):
                cx = cx0 + oi * opt_step * pw
                page.draw_circle(fitz.Point(cx, cy), bub_r_pt, color=BUB_CLR, width=0.85)
                page.insert_text(fitz.Point(cx - 2.2, cy + 2.3), opt, fontsize=6.0, fontname="hebo", color=BUB_CLR)
                if interactive:
                    w = fitz.Widget()
                    w.field_name = f"Q{q}_{opt}"
                    w.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
                    w.field_value = "Off"
                    w.rect = fitz.Rect(cx - bub_r_pt, cy - bub_r_pt, cx + bub_r_pt, cy + bub_r_pt)
                    page.add_widget(w)

        # ── 6. FOOTER ──────────────────────────────────────────────────────────
        foot_y = ph - m - 8.0
        foot_text = f"Page {page_idx + 1} of {total_pages}   |   Questions {p_start}–{p_end} of {count}"
        page.draw_line(fitz.Point(L, foot_y - 8), fitz.Point(R, foot_y - 8), color=LINE_CLR, width=0.5)
        page.insert_text(fitz.Point(L + CW/2 - len(foot_text)*2.2, foot_y), foot_text, fontsize=7.2, fontname="helv", color=GRAY)

    return doc

# Pre-calibrated seed templates whose coordinates align exactly with the PDF sheets
# generated by generate_omr_pdf. Multi-page sheets (150Q, 200Q) paginate automatically.
SEED_TEMPLATES = [
    {
        "name": "Standard 100Q (4 cols x 25)",
        "config": {
            "question_count": 100, "options": ["A", "B", "C", "D"],
            "page_number": 1, "layout_mode": "grid",
            "x_start": 0.136, "y_start": 0.238,
            "x_step": 0.0433, "y_step": 0.0248,
            "questions_per_column": 25, "option_step": 0.0415,
            "bubble_radius": 8, "max_cols_per_page": 4,
            "darkness_threshold": 90, "min_fill_ratio": 0.18, "max_fill_ratio": 0.70,
        },
    },
    {
        "name": "Standard 150Q (2 pages, 100Q + 50Q)",
        "config": {
            "question_count": 150, "options": ["A", "B", "C", "D"],
            "page_number": 1, "layout_mode": "grid",
            "x_start": 0.136, "y_start": 0.238,
            "x_step": 0.0433, "y_step": 0.0248,
            "questions_per_column": 25, "option_step": 0.0415,
            "bubble_radius": 8, "max_cols_per_page": 4,
            "darkness_threshold": 90, "min_fill_ratio": 0.18, "max_fill_ratio": 0.70,
        },
    },
    {
        "name": "Standard 60Q (3 cols x 20)",
        "config": {
            "question_count": 60, "options": ["A", "B", "C", "D"],
            "page_number": 1, "layout_mode": "grid",
            "x_start": 0.144, "y_start": 0.238,
            "x_step": 0.0577, "y_step": 0.0248,
            "questions_per_column": 20, "option_step": 0.0595,
            "bubble_radius": 8, "max_cols_per_page": 4,
            "darkness_threshold": 90, "min_fill_ratio": 0.18, "max_fill_ratio": 0.70,
        },
    },
    {
        "name": "Standard 50Q (2 cols x 25)",
        "config": {
            "question_count": 50, "options": ["A", "B", "C", "D"],
            "page_number": 1, "layout_mode": "grid",
            "x_start": 0.162, "y_start": 0.238,
            "x_step": 0.0866, "y_step": 0.0248,
            "questions_per_column": 25, "option_step": 0.0956,
            "bubble_radius": 8, "max_cols_per_page": 4,
            "darkness_threshold": 90, "min_fill_ratio": 0.18, "max_fill_ratio": 0.70,
        },
    },
    {
        "name": "Standard 200Q (2 pages, 100Q + 100Q)",
        "config": {
            "question_count": 200, "options": ["A", "B", "C", "D"],
            "page_number": 1, "layout_mode": "grid",
            "x_start": 0.136, "y_start": 0.238,
            "x_step": 0.0433, "y_step": 0.0248,
            "questions_per_column": 25, "option_step": 0.0415,
            "bubble_radius": 8, "max_cols_per_page": 4,
            "darkness_threshold": 90, "min_fill_ratio": 0.18, "max_fill_ratio": 0.70,
        },
    },
]

@app.route("/templates/seed", methods=["POST"])
@login_required
def seed_templates():
    uid = session["user_id"]
    added = 0
    with db() as con:
        existing_names = {row["name"] for row in
                          con.execute("SELECT name FROM templates WHERE user_id=?", (uid,)).fetchall()}
        for t in SEED_TEMPLATES:
            if t["name"] not in existing_names:
                con.execute(
                    "INSERT INTO templates(user_id,name,config_json,created_at) VALUES(?,?,?,?)",
                    (uid, t["name"], json.dumps(t["config"]),
                     datetime.utcnow().isoformat(timespec="seconds"))
                )
                added += 1
    if added:
        flash(f"Added {added} standard template(s). Download the matching answer sheets from your Dashboard.", "success")
    else:
        flash("All standard templates are already in your workspace.", "success")
    return redirect(url_for("dashboard"))

@app.route("/")
def index():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return render_template("landing.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if len(username) < 3 or len(password) < 8 or "@" not in email:
            flash("Use a username of at least 3 characters, a valid email, and a password of at least 8 characters.", "error")
            return render_template("register.html")
        try:
            with db() as con:
                con.execute("INSERT INTO users(username,email,password_hash,created_at) VALUES(?,?,?,?)",
                            (username, email, hash_password(password), datetime.utcnow().isoformat(timespec="seconds")))
            flash("Account created. Please sign in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("That username or email is already registered.", "error")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        ident = request.form.get("identity", "").strip().lower()
        password = request.form.get("password", "")
        with db() as con:
            user = con.execute("SELECT * FROM users WHERE lower(username)=? OR email=?", (ident, ident)).fetchone()
        if user and verify_password(password, user["password_hash"]):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        flash("Incorrect username/email or password.", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    uid = session["user_id"]
    with db() as con:
        raw_templates = con.execute("SELECT * FROM templates WHERE user_id=? ORDER BY id DESC", (uid,)).fetchall()
        results = con.execute("SELECT * FROM results WHERE user_id=? ORDER BY id DESC LIMIT 10", (uid,)).fetchall()
        total = con.execute("SELECT COUNT(*) n FROM results WHERE user_id=?", (uid,)).fetchone()["n"]

    templates = []
    for t in raw_templates:
        item = dict(t)
        try:
            item["cfg"] = json.loads(t["config_json"])
        except Exception:
            item["cfg"] = {}
        templates.append(item)

    return render_template("dashboard.html", templates=templates, results=results, total=total)

@app.route("/templates/<int:template_id>/pdf")
@login_required
def template_pdf(template_id):
    template = get_user_template(template_id, session["user_id"])
    if not template:
        abort(404)
    cfg = json.loads(template["config_json"])
    interactive = request.args.get("interactive", "0") in ("1", "true", "yes")
    doc = generate_omr_pdf(template["name"], cfg, interactive=interactive)
    pdf_bytes = doc.tobytes()
    doc.close()

    safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", template["name"].strip()).strip("_") or "omr_template"
    suffix = "fillable" if interactive else "sheet"
    filename = f"{safe_name}_{suffix}.pdf"

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename
    )

@app.route("/templates/<int:template_id>/json")
@login_required
def template_json(template_id):
    template = get_user_template(template_id, session["user_id"])
    if not template:
        abort(404)
    cfg = json.loads(template["config_json"])
    data = {
        "id": template["id"],
        "name": template["name"],
        "created_at": template["created_at"],
        "config": cfg
    }
    json_str = json.dumps(data, indent=2)
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", template["name"].strip()).strip("_") or "omr_template"
    return send_file(
        io.BytesIO(json_str.encode("utf-8")),
        mimetype="application/json",
        as_attachment=True,
        download_name=f"{safe_name}_config.json"
    )

@app.route("/templates/sample-pdf")
def sample_pdf():
    count = request.args.get("count", 100, type=int)
    if count not in (30, 50, 60, 100, 120, 125, 150, 200):
        count = 100
    interactive = request.args.get("interactive", "0") in ("1", "true", "yes")

    # Match pre-calibrated seed template configuration
    matching_seed = next((t["config"] for t in SEED_TEMPLATES if t["config"]["question_count"] == count), None)
    if matching_seed:
        cfg = dict(matching_seed)
    else:
        cfg = {
            "question_count": count,
            "options": ["A", "B", "C", "D"],
            "page_number": 1,
            "layout_mode": "grid",
            "x_start": 0.136,
            "y_start": 0.238,
            "x_step": 0.0433,
            "y_step": 0.0248,
            "questions_per_column": 25,
            "option_step": 0.0415,
            "bubble_radius": 8,
            "max_cols_per_page": 4,
            "darkness_threshold": 90,
            "min_fill_ratio": 0.18,
            "max_fill_ratio": 0.70
        }
    doc = generate_omr_pdf(f"SAMPLE_{count}Q_OMR", cfg, interactive=interactive)
    pdf_bytes = doc.tobytes()
    doc.close()
    suffix = "fillable" if interactive else "sheet"
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"sample_{count}q_omr_{suffix}.pdf"
    )

@app.route("/templates/<int:template_id>/sample-key")
@login_required
def template_sample_key(template_id):
    t = get_user_template(template_id, session["user_id"])
    if not t:
        abort(404)
    cfg = json.loads(t["config_json"])
    count = cfg.get("question_count", 100)
    opts = cfg.get("options", ["A", "B", "C", "D"])
    answers = [opts[i % len(opts)] for i in range(count)]
    lines = [
        f"# OMR Answer Key for: {t['name']}",
        f"# Total Questions: {count}",
        f"# Options: {', '.join(opts)}",
        "# You can paste this text directly into the 'Answer key' field when scanning.",
        "",
        ", ".join(answers),
        "",
        "# Numbered format alternative:",
        *(f"{i+1}. {ans}" for i, ans in enumerate(answers))
    ]
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", t["name"].strip()).strip("_")
    return send_file(
        io.BytesIO("\n".join(lines).encode("utf-8")),
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"{safe_name}_answer_key_{count}q.txt"
    )

@app.route("/sample-key")
def download_sample_key():
    count = request.args.get("count", 100, type=int)
    if count not in (30, 50, 60, 100, 120, 125, 150, 200):
        count = 100
    opts = ["A", "B", "C", "D"]
    answers = [opts[i % len(opts)] for i in range(count)]
    lines = [
        f"# Sample OMR Answer Key ({count} Questions)",
        "# Valid Options: A, B, C, D",
        "# Format: comma-separated or numbered lines",
        "",
        ", ".join(answers),
        "",
        "# Numbered format alternative:",
        *(f"{i+1}. {ans}" for i, ans in enumerate(answers))
    ]
    return send_file(
        io.BytesIO("\n".join(lines).encode("utf-8")),
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"sample_answer_key_{count}q.txt"
    )

def parse_answer_key(raw_text):
    """Parse raw answer key supporting comma-separated, space-separated, newlines,
    and numbered lines like '1. A', '1: A', 'Q1 - A', 'Q1=A'."""
    raw = (raw_text or "").strip()
    if not raw:
        return []
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("#")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\b(?:q\s*)?\d+[\.\:\-\)\=]\s*", " ", cleaned, flags=re.I)
    tokens = [x.strip().upper() for x in re.split(r"[,\s]+", cleaned) if x.strip()]
    return tokens

@app.route("/templates/new", methods=["GET", "POST"])
@login_required
def new_template():
    if request.method == "POST":
        try:
            cfg = normalize_config(request.form)
            name = request.form.get("name", "").strip() or "Custom OMR Template"
            with db() as con:
                con.execute("INSERT INTO templates(user_id,name,config_json,created_at) VALUES(?,?,?,?)",
                            (session["user_id"], name, json.dumps(cfg), datetime.utcnow().isoformat(timespec="seconds")))
            flash("Template saved! You can now download your printable OMR sheet or score a completed test.", "success")
            return redirect(url_for("dashboard"))
        except (ValueError, TypeError) as e:
            flash(str(e), "error")
    return render_template("template_form.html")

@app.route("/templates/<int:template_id>/delete", methods=["POST"])
@login_required
def delete_template(template_id):
    with db() as con:
        con.execute("DELETE FROM templates WHERE id=? AND user_id=?", (template_id, session["user_id"]))
    flash("Template deleted.", "success")
    return redirect(url_for("dashboard"))

@app.route("/scan", methods=["GET", "POST"])
@login_required
def scan():
    uid = session["user_id"]
    with db() as con:
        raw_templates = con.execute("SELECT * FROM templates WHERE user_id=? ORDER BY name", (uid,)).fetchall()

    templates = []
    for t in raw_templates:
        item = dict(t)
        try:
            item["cfg"] = json.loads(t["config_json"])
        except Exception:
            item["cfg"] = {}
        templates.append(item)

    selected_template_id = request.args.get("template_id", type=int)

    if request.method == "POST":
        template_id = request.form.get("template_id", type=int)
        selected_template_id = template_id
        template = get_user_template(template_id, uid) if template_id else None
        if not template:
            flash("Choose one of your saved templates.", "error")
            return render_template("scan.html", templates=templates, selected_template_id=selected_template_id)
        upload = request.files.get("pdf")
        if not upload or not upload.filename or not allowed_file(upload.filename):
            flash("Please upload a PDF file.", "error")
            return render_template("scan.html", templates=templates, selected_template_id=selected_template_id)
        filename = secure_filename(upload.filename)
        unique_name = f"{secrets.token_hex(8)}_{filename}"
        path = os.path.join(UPLOAD_DIR, unique_name)
        upload.save(path)
        try:
            # Verify PDF and cap pages / malformed files.
            doc = fitz.open(path)
            if len(doc) > 100:
                doc.close()
                raise ValueError("PDF has too many pages (maximum 100).")
            doc.close()
            cfg = json.loads(template["config_json"])
            detected, diagnostics, method = scan_answers(path, cfg)
            key_raw = request.form.get("answer_key", "")
            key_tokens = parse_answer_key(key_raw)
            if len(key_tokens) != cfg["question_count"]:
                raise ValueError(
                    f"Answer key contains {len(key_tokens)} answers, but this template requires exactly {cfg['question_count']}. "
                    f"Tip: Use the '⚡ Fill Sample Key' button on the scan page to automatically generate a matching key."
                )
            invalid_opts = [x for x in key_tokens if x not in cfg["options"]]
            if invalid_opts:
                raise ValueError(
                    f"Answer key contains invalid option '{invalid_opts[0]}'. "
                    f"Allowed options for this template are: {', '.join(cfg['options'])}."
                )
            key = {i+1: ans for i, ans in enumerate(key_tokens)}
            details = []
            correct = incorrect = unanswered = 0
            for q in range(1, cfg["question_count"]+1):
                given = detected.get(q)
                expected = key[q]
                if not given:
                    status = "Unanswered / review"
                    unanswered += 1
                elif given == expected:
                    status = "Correct"
                    correct += 1
                else:
                    status = "Incorrect"
                    incorrect += 1
                details.append({"question": q, "marked": given or "", "correct": expected, "status": status,
                                "diagnostics": diagnostics.get(q, {})})
            score = float(correct)  # requested scheme: +1 correct, no negative marking
            test_name = request.form.get("test_name", "").strip() or "OMR Test"
            with db() as con:
                cur = con.execute("""INSERT INTO results(user_id,template_id,test_name,filename,question_count,correct_count,
                  incorrect_count,unanswered_count,score,details_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                  (uid, template_id, test_name, filename, cfg["question_count"], correct, incorrect, unanswered,
                   score, json.dumps({"details": details, "method": method, "template": template["name"]}),
                   datetime.utcnow().isoformat(timespec="seconds")))
                result_id = cur.lastrowid
            return redirect(url_for("result_detail", result_id=result_id))
        except Exception as e:
            try:
                os.remove(path)
            except OSError:
                pass
            flash(f"Could not score this PDF: {e}", "error")
    return render_template("scan.html", templates=templates, selected_template_id=selected_template_id)

@app.route("/results/<int:result_id>")
@login_required
def result_detail(result_id):
    with db() as con:
        result = con.execute("SELECT * FROM results WHERE id=? AND user_id=?", (result_id, session["user_id"])).fetchone()
    if not result:
        abort(404)
    payload = json.loads(result["details_json"])
    pct = round((result["correct_count"] / result["question_count"] * 100), 1) if result["question_count"] else 0
    return render_template("result.html", result=result, details=payload["details"], method=payload.get("method"), pct=pct,
                           template_name=payload.get("template", "Template"))

@app.route("/results/<int:result_id>/csv")
@login_required
def result_csv(result_id):
    with db() as con:
        result = con.execute("SELECT * FROM results WHERE id=? AND user_id=?", (result_id, session["user_id"])).fetchone()
    if not result:
        abort(404)
    payload = json.loads(result["details_json"])
    import csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Question", "Your Answer", "Correct Answer", "Status", "Detection Confidence", "Detection Method"])
    for row in payload["details"]:
        diag = row.get("diagnostics", {})
        writer.writerow([row["question"], row["marked"], row["correct"], row["status"], diag.get("confidence", ""), diag.get("method", "")])
    mem = io.BytesIO(output.getvalue().encode("utf-8-sig"))
    mem.seek(0)
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name=f"omr_result_{result_id}.csv")

@app.route("/results/<int:result_id>/delete", methods=["POST"])
@login_required
def delete_result(result_id):
    with db() as con:
        con.execute("DELETE FROM results WHERE id=? AND user_id=?", (result_id, session["user_id"]))
    flash("Result deleted.", "success")
    return redirect(url_for("dashboard"))

with app.app_context():
    init_db()

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
