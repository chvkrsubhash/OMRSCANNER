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
        "x_start": float(form.get("x_start", "0.1176")),
        "y_start": float(form.get("y_start", "0.3658")),
        "x_step": float(form.get("x_step", "0.0433")),
        "y_step": float(form.get("y_step", "0.0226")),
        "questions_per_column": int(form.get("questions_per_column", str(count))),
        "option_step": float(form.get("option_step", "0.0311")),
        "bubble_radius": int(form.get("bubble_radius", "10")),
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
    Features pink border header box, candidate lines, 6 roll number write-in boxes with 0-9 bubble circle columns,
    and automatic multi-page generation when questions exceed 100 so bubbles never overlap page borders."""
    doc = fitz.open()
    pw, ph = 595.3, 841.9  # Standard A4 dimensions in points
    m = 24.0
    L = m + 16.0
    R = pw - m - 16.0
    CW = R - L

    PINK = (0.89, 0.18, 0.42)
    PINK_LINE = (0.94, 0.55, 0.70)
    DKINK = (0.12, 0.13, 0.18)
    GRAY = (0.45, 0.48, 0.55)
    LGRAY = (0.85, 0.86, 0.88)

    options = cfg.get("options", ["A", "B", "C", "D"])
    count = cfg.get("question_count", 100)
    qpc = max(1, cfg.get("questions_per_column", 25))
    max_cols = max(1, cfg.get("max_cols_per_page", 4))
    q_per_page = max_cols * qpc
    total_pages = max(1, (count + q_per_page - 1) // q_per_page)

    col_w = CW / float(max_cols)
    bub_r_pt = 4.8
    opt_gap = 18.5
    y_start_pt = 308.0
    y_step_pt = 19.0

    title_text = (template_name or "APPSC PRACTICE TEST").replace("_", " ").strip().upper()

    for page_idx in range(total_pages):
        page = doc.new_page(width=pw, height=ph)

        # 1. Corner fiducial markers (solid black)
        marker_size = 11.0
        page.draw_rect(fitz.Rect(m, m, m + marker_size, m + marker_size), color=(0,0,0), fill=(0,0,0))
        page.draw_rect(fitz.Rect(pw - m - marker_size, m, pw - m, m + marker_size), color=(0,0,0), fill=(0,0,0))
        page.draw_rect(fitz.Rect(m, ph - m - marker_size, m, ph - m), color=(0,0,0), fill=(0,0,0))
        page.draw_rect(fitz.Rect(pw - m - marker_size, ph - m - marker_size, pw - m, ph - m), color=(0,0,0), fill=(0,0,0))

        # 2. Timing track down left margin
        for ty in range(int(m + 80), int(ph - m - 30), 18):
            page.draw_rect(fitz.Rect(m, ty, m + 6.0, ty + 5.0), color=(0,0,0), fill=(0,0,0))

        p_start = page_idx * q_per_page + 1
        p_end = min(count, (page_idx + 1) * q_per_page)

        # ── 3. HEADER BOX WITH PINK BORDER ──────────────────────────────────────
        h_top = m + 10.0
        h_bot = h_top + 46.0
        page.draw_rect(fitz.Rect(L, h_top, R, h_bot), color=PINK, width=1.4)

        page.insert_text(fitz.Point(L + CW/2 - len(title_text)*3.8, h_top + 22), title_text, fontsize=14, fontname="hebo", color=PINK)
        sub_text = "Daily Test" if page_idx == 0 else f"Daily Test · Page {page_idx + 1} of {total_pages}"
        page.insert_text(fitz.Point(L + CW/2 - len(sub_text)*2.4, h_top + 37), sub_text, fontsize=8.5, fontname="helv", color=GRAY)
        page.insert_text(fitz.Point(R - 64, h_top + 16), "Booklet A", fontsize=9, fontname="hebo", color=(0,0,0))

        # ── 4. CANDIDATE INFO (LEFT SIDE) ───────────────────────────────────────
        info_w = CW * 0.52
        cand_fields = [
            ("Candidate Name:", f"CandidateName_P{page_idx + 1}"),
            ("Roll No.:",       f"RollNoText_P{page_idx + 1}"),
            ("Class / Section:",f"ClassSection_P{page_idx + 1}"),
            ("Date:",           f"Date_P{page_idx + 1}"),
            ("Signature:",      f"Signature_P{page_idx + 1}"),
        ]
        fy0 = h_bot + 24.0
        fgap = 18.0
        for i, (lbl, fld_name) in enumerate(cand_fields):
            fy = fy0 + i * fgap
            page.insert_text(fitz.Point(L, fy), lbl, fontsize=8.5, fontname="helv", color=DKINK)
            page.draw_line(fitz.Point(L + 76, fy + 3), fitz.Point(L + info_w - 10, fy + 3), color=PINK_LINE, width=1.0)
            if interactive:
                w = fitz.Widget()
                w.field_name = fld_name
                w.field_type = fitz.PDF_WIDGET_TYPE_TEXT
                w.rect = fitz.Rect(L + 76, fy - 10, L + info_w - 10, fy + 3)
                page.add_widget(w)

        # ── 5. ROLL NUMBER (6 BOXES + 0-9 BUBBLE CIRCLES) ──────────────────────
        rn_x0 = L + CW * 0.56
        page.insert_text(fitz.Point(rn_x0, h_bot + 14.0), "Roll Number", fontsize=8.5, fontname="hebo", color=(0,0,0))

        NUM_DIGITS = 6
        BOX_W = 16.5
        BOX_H = 15.0
        BOX_GAP = 5.0
        box_y0 = h_bot + 22.0

        for d in range(NUM_DIGITS):
            bx = rn_x0 + d * (BOX_W + BOX_GAP)
            page.draw_rect(fitz.Rect(bx, box_y0, bx + BOX_W, box_y0 + BOX_H), color=PINK, width=1.0, fill=(1,1,1))
            if interactive:
                w = fitz.Widget()
                w.field_name = f"RollDigit_P{page_idx + 1}_{d + 1}"
                w.field_type = fitz.PDF_WIDGET_TYPE_TEXT
                w.rect = fitz.Rect(bx, box_y0, bx + BOX_W, box_y0 + BOX_H)
                page.add_widget(w)

        # 0-9 vertical bubble columns under each digit box
        DBUB_R = 4.8
        DBUB_Y0 = box_y0 + BOX_H + 8.0
        DBUB_STP = 11.5
        for d in range(NUM_DIGITS):
            col_cx = rn_x0 + d * (BOX_W + BOX_GAP) + BOX_W / 2.0
            for digit in range(10):
                cy = DBUB_Y0 + digit * DBUB_STP
                page.draw_circle(fitz.Point(col_cx, cy), DBUB_R, color=PINK, width=0.85)
                page.insert_text(fitz.Point(col_cx - 2.0, cy + 2.5), str(digit), fontsize=6.2, fontname="helv", color=PINK)
                if interactive:
                    w = fitz.Widget()
                    w.field_name = f"RollDigit_P{page_idx + 1}_{d + 1}_{digit}"
                    w.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
                    w.field_value = "Off"
                    w.rect = fitz.Rect(col_cx - DBUB_R, cy - DBUB_R, col_cx + DBUB_R, cy + DBUB_R)
                    page.add_widget(w)

        # ── 6. INSTRUCTIONS ────────────────────────────────────────────────────
        instr_y = DBUB_Y0 + 9 * DBUB_STP + 22.0
        page.insert_text(fitz.Point(L, instr_y), "Instructions:", fontsize=8.5, fontname="hebo", color=DKINK)
        page.insert_text(fitz.Point(L + 60, instr_y), "Use a blue/black ball pen. Darken one bubble completely. Do not make stray marks.", fontsize=7.8, fontname="helv", color=DKINK)

        # ── 7. QUESTION ROWS (UP TO 4 COLUMNS) ─────────────────────────────────
        for q in range(p_start, p_end + 1):
            local_idx = q - p_start
            col = local_idx // qpc
            row = local_idx % qpc

            cx0 = L + col * col_w + 30.0
            cy = y_start_pt + row * y_step_pt

            page.insert_text(fitz.Point(cx0 - bub_r_pt - 18, cy + 2.8), f"{q}.", fontsize=7.8, fontname="helv", color=DKINK)
            for oi, opt in enumerate(options):
                cx = cx0 + oi * opt_gap
                page.draw_circle(fitz.Point(cx, cy), bub_r_pt, color=PINK, width=0.85)
                page.insert_text(fitz.Point(cx - 2.3, cy + 2.5), opt, fontsize=6.2, fontname="helv", color=PINK)
                if interactive:
                    w = fitz.Widget()
                    w.field_name = f"Q{q}_{opt}"
                    w.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
                    w.field_value = "Off"
                    w.rect = fitz.Rect(cx - bub_r_pt, cy - bub_r_pt, cx + bub_r_pt, cy + bub_r_pt)
                    page.add_widget(w)

        # ── 8. FOOTER ──────────────────────────────────────────────────────────
        foot_y = ph - m - 8.0
        foot_text = f"Page {page_idx + 1} of {total_pages}   |   Questions {p_start}–{p_end} of {count}"
        page.draw_line(fitz.Point(L, foot_y - 8), fitz.Point(R, foot_y - 8), color=LGRAY, width=0.6)
        page.insert_text(fitz.Point(L + CW/2 - len(foot_text)*2.2, foot_y), foot_text, fontsize=7.2, fontname="helv", color=GRAY)

    return doc

# Pre-calibrated seed templates calibrated to align exactly with the pink exam format
# with 6 roll number boxes and 0-9 circle grid. Multi-page sheets paginate automatically.
SEED_TEMPLATES = [
    {
        "name": "Standard 100Q (4 cols x 25)",
        "config": {
            "question_count": 100, "options": ["A", "B", "C", "D"],
            "page_number": 1, "layout_mode": "grid",
            "x_start": 0.1176, "y_start": 0.3658,
            "x_step": 0.0433, "y_step": 0.0226,
            "questions_per_column": 25, "option_step": 0.0311,
            "bubble_radius": 10, "max_cols_per_page": 4,
            "darkness_threshold": 90, "min_fill_ratio": 0.18, "max_fill_ratio": 0.70,
        },
    },
    {
        "name": "Standard 150Q (2 pages, 100Q + 50Q)",
        "config": {
            "question_count": 150, "options": ["A", "B", "C", "D"],
            "page_number": 1, "layout_mode": "grid",
            "x_start": 0.1176, "y_start": 0.3658,
            "x_step": 0.0433, "y_step": 0.0226,
            "questions_per_column": 25, "option_step": 0.0311,
            "bubble_radius": 10, "max_cols_per_page": 4,
            "darkness_threshold": 90, "min_fill_ratio": 0.18, "max_fill_ratio": 0.70,
        },
    },
    {
        "name": "Standard 50Q (2 cols x 25)",
        "config": {
            "question_count": 50, "options": ["A", "B", "C", "D"],
            "page_number": 1, "layout_mode": "grid",
            "x_start": 0.1176, "y_start": 0.3658,
            "x_step": 0.0433, "y_step": 0.0226,
            "questions_per_column": 25, "option_step": 0.0311,
            "bubble_radius": 10, "max_cols_per_page": 4,
            "darkness_threshold": 90, "min_fill_ratio": 0.18, "max_fill_ratio": 0.70,
        },
    },
    {
        "name": "Standard 60Q (3 cols x 20)",
        "config": {
            "question_count": 60, "options": ["A", "B", "C", "D"],
            "page_number": 1, "layout_mode": "grid",
            "x_start": 0.1176, "y_start": 0.3658,
            "x_step": 0.0433, "y_step": 0.0261,
            "questions_per_column": 20, "option_step": 0.0311,
            "bubble_radius": 10, "max_cols_per_page": 4,
            "darkness_threshold": 90, "min_fill_ratio": 0.18, "max_fill_ratio": 0.70,
        },
    },
    {
        "name": "Standard 200Q (2 pages, 100Q + 100Q)",
        "config": {
            "question_count": 200, "options": ["A", "B", "C", "D"],
            "page_number": 1, "layout_mode": "grid",
            "x_start": 0.1176, "y_start": 0.3658,
            "x_step": 0.0433, "y_step": 0.0226,
            "questions_per_column": 25, "option_step": 0.0311,
            "bubble_radius": 10, "max_cols_per_page": 4,
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
            "x_start": 0.1176,
            "y_start": 0.3658,
            "x_step": 0.0433,
            "y_step": 0.0226,
            "questions_per_column": 25,
            "option_step": 0.0311,
            "bubble_radius": 10,
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
