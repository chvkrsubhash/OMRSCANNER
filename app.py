import os, json, sqlite3, hashlib, secrets, re
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, abort, send_from_directory
from werkzeug.utils import secure_filename
import fitz
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from dotenv import load_dotenv
import boto3
import requests

load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("VERCEL"):
    DATA_DIR = os.path.join("/tmp", "instance")
else:
    DATA_DIR = os.path.join(APP_DIR, "instance")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "omr.sqlite3")
ALLOWED_EXTENSIONS = {"pdf"}

# AWS S3 Storage Configuration
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "")
AWS_S3_PREFIX = os.environ.get("AWS_S3_PREFIX", "weekly-rough-work").strip("/")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

_s3_client = None

def get_s3_client():
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    if AWS_S3_BUCKET and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        try:
            _s3_client = boto3.client(
                "s3",
                region_name=AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            )
            return _s3_client
        except Exception as e:
            print(f"[S3 Init Error] {e}")
    return None

def s3_upload_file(local_path_or_bytes, s3_subpath, content_type="application/pdf"):
    """Upload a file or raw bytes to S3 bucket under AWS_S3_PREFIX with local fallback."""
    client = get_s3_client()
    if not client or not AWS_S3_BUCKET:
        return None
    key = f"{AWS_S3_PREFIX}/{s3_subpath.lstrip('/')}"
    try:
        if isinstance(local_path_or_bytes, (bytes, bytearray)):
            client.put_object(Bucket=AWS_S3_BUCKET, Key=key, Body=local_path_or_bytes, ContentType=content_type)
        else:
            with open(local_path_or_bytes, "rb") as f:
                client.put_object(Bucket=AWS_S3_BUCKET, Key=key, Body=f.read(), ContentType=content_type)
        return key
    except Exception as e:
        print(f"[S3 Upload Notice] {key}: {e}")
        return None

def s3_download_file(s3_subpath):
    """Download a file's raw bytes from S3 bucket under AWS_S3_PREFIX."""
    client = get_s3_client()
    if not client or not AWS_S3_BUCKET:
        return None
    key = f"{AWS_S3_PREFIX}/{s3_subpath.lstrip('/')}"
    try:
        res = client.get_object(Bucket=AWS_S3_BUCKET, Key=key)
        return res["Body"].read()
    except Exception as e:
        return None

def sync_db_from_s3():
    """Restore database from S3 backup if local database is missing (useful for Vercel cold starts)."""
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        db_bytes = s3_download_file("omr-db/omr.sqlite3")
        if db_bytes:
            try:
                os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
                with open(DB_PATH, "wb") as f:
                    f.write(db_bytes)
                print("[S3 Sync] Restored omr.sqlite3 from S3!")
            except Exception as e:
                print(f"[S3 Sync Warning] {e}")

def sync_db_to_s3():
    """Backup database to S3 so changes persist across serverless restarts."""
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
        try:
            with open(DB_PATH, "rb") as f:
                s3_upload_file(f.read(), "omr-db/omr.sqlite3", content_type="application/x-sqlite3")
        except Exception as e:
            pass

def sync_result_to_firestore(result_data):
    """Sync scan evaluation results to Firebase Firestore via REST API if configured."""
    project_id = os.environ.get("FIREBASE_PROJECT_ID")
    api_key = os.environ.get("FIREBASE_API_KEY")
    if not project_id or not api_key:
        return
    try:
        url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/omr_evaluations?key={api_key}"
        doc_fields = {
            "test_name": {"stringValue": str(result_data.get("test_name", ""))},
            "candidate_name": {"stringValue": str(result_data.get("candidate_name", ""))},
            "roll_number": {"stringValue": str(result_data.get("roll_number", ""))},
            "score": {"doubleValue": float(result_data.get("score", 0.0))},
            "question_count": {"integerValue": str(result_data.get("question_count", 0))},
            "correct_count": {"integerValue": str(result_data.get("correct_count", 0))},
            "incorrect_count": {"integerValue": str(result_data.get("incorrect_count", 0))},
            "unanswered_count": {"integerValue": str(result_data.get("unanswered_count", 0))},
            "created_at": {"stringValue": str(result_data.get("created_at", datetime.utcnow().isoformat()))}
        }
        requests.post(url, json={"fields": doc_fields}, timeout=2.0)
    except Exception:
        pass

app = Flask(
    __name__,
    static_folder=os.path.join(APP_DIR, "static"),
    static_url_path="/static",
    template_folder=os.path.join(APP_DIR, "templates")
)
app.secret_key = os.environ.get("SECRET_KEY", "replace-this-with-a-long-random-secret")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB for batch uploads

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(os.path.join(APP_DIR, "static"), filename)

@app.context_processor
def inject_firebase():
    return {
        "FIREBASE_CONFIG": {
            "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
            "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
            "projectId": os.environ.get("FIREBASE_PROJECT_ID", ""),
            "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
            "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
            "appId": os.environ.get("FIREBASE_APP_ID", "")
        }
    }

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    sync_db_from_s3()
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
          created_at TEXT NOT NULL,
          batch_id TEXT,
          roll_number TEXT,
          candidate_name TEXT,
          stored_filename TEXT
        );
        """)
        # Safe migration for existing tables
        cols = [r[1] for r in con.execute("PRAGMA table_info(results)").fetchall()]
        for col_name in ("batch_id", "roll_number", "candidate_name", "stored_filename"):
            if col_name not in cols:
                try:
                    con.execute(f"ALTER TABLE results ADD COLUMN {col_name} TEXT")
                except Exception:
                    pass

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

def render_pdf_page_rgb(pdf_path, page_number):
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
    return img

def render_pdf_page(pdf_path, page_number):
    img = render_pdf_page_rgb(pdf_path, page_number)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def compute_ink_mask(rgb_img):
    """Detect ink from blue ball/gel pen, black pen, pencil, or markers on white/light paper."""
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    # Detect ink: black/dark ink or blue/colored ink vs white paper
    return (gray < 165) | ((hsv[:, :, 1] > 50) & (gray < 225))

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
                candidates = [value.upper(), name.upper().split("_")[-1]]
                selected = next((c for c in candidates if c in options), None)
                if selected:
                    answers[q] = selected
    finally:
        doc.close()
    return answers

def scan_roll_number_visual(rgb_img):
    """Visually detect the 6-digit roll number from the vertical 0-9 bubble columns."""
    h, w, _ = rgb_img.shape
    is_ink = compute_ink_mask(rgb_img)

    NUM_DIGITS = 6
    detected_digits = []

    # Standard normalized coordinates for 6 roll number bubble columns:
    # rn_x0 = 328.568 pt / 595.3 = 0.551936
    # col_step = 21.5 pt / 595.3 = 0.036116
    # col_half_w = (16.5 / 2.0) / 595.3 = 0.013858
    # y0 = 125.0 pt / 841.9 = 0.148474
    # y_step = 11.5 pt / 841.9 = 0.013659
    # bub_r = 4.8 pt / 595.3 = 0.008063
    bub_r_px = max(4, int(0.008063 * w))

    for d in range(NUM_DIGITS):
        col_cx_norm = 0.551936 + d * 0.036116 + 0.013858
        cx = int(col_cx_norm * w)

        digit_ratios = {}
        for digit in range(10):
            cy_norm = 0.148474 + digit * 0.013659
            cy = int(cy_norm * h)

            y1, y2 = max(0, cy - bub_r_px), min(h, cy + bub_r_px + 1)
            x1, x2 = max(0, cx - bub_r_px), min(w, cx + bub_r_px + 1)
            roi_ink = is_ink[y1:y2, x1:x2]
            if roi_ink.size == 0:
                digit_ratios[digit] = 0.0
                continue

            yy, xx = np.ogrid[:roi_ink.shape[0], :roi_ink.shape[1]]
            mask = (xx - (roi_ink.shape[1]-1)/2)**2 + (yy - (roi_ink.shape[0]-1)/2)**2 <= (bub_r_px * 0.72)**2
            vals = roi_ink[mask]
            digit_ratios[digit] = float(np.mean(vals)) if vals.size else 0.0

        ranked = sorted(digit_ratios.items(), key=lambda kv: kv[1], reverse=True)
        top_digit, top_ratio = ranked[0]
        second_ratio = ranked[1][1] if len(ranked) > 1 else 0.0

        if top_ratio >= 0.45 or (top_ratio >= 0.25 and top_ratio - second_ratio >= 0.10):
            detected_digits.append(str(top_digit))
        else:
            detected_digits.append("")

    res = "".join(detected_digits).strip()
    return res if len(res) >= 3 else ""

def scan_answers(pdf_path, cfg):
    options = cfg["options"]
    count = cfg["question_count"]

    # Interactive AcroForm fields are more reliable than visual detection if usable fields exist.
    form_answers = read_acroform_answers(pdf_path, count, options)
    if len(form_answers) >= max(1, int(count * 0.5)):
        return form_answers, {q: {"method": "pdf_form", "confidence": 1.0, "ratios": {}} for q in form_answers}, "pdf_form"

    # Multi-page visual scanning support:
    # Auto-calibrate if old/custom template coordinates deviate from standard pink sheet
    if cfg.get("y_start", 0) < 0.33 or cfg.get("y_start", 0) > 0.45:
        cfg["y_start"] = 0.3658
        cfg["x_start"] = 0.1176
        cfg["x_step"] = 0.0433
        cfg["y_step"] = 0.0226
        cfg["option_step"] = 0.0311
        cfg["bubble_radius"] = 10

    qpc = max(1, cfg.get("questions_per_column", 25))
    max_cols = max(1, cfg.get("max_cols_per_page", 4))
    q_per_page = max_cols * qpc

    doc = fitz.open(pdf_path)
    total_doc_pages = len(doc)
    doc.close()

    base_page = max(1, cfg.get("page_number", 1))
    page_cache = {}

    def get_page_data(p_num):
        if p_num not in page_cache:
            if p_num <= total_doc_pages:
                rgb = render_pdf_page_rgb(pdf_path, p_num)
                is_ink = compute_ink_mask(rgb)
                page_cache[p_num] = (rgb, is_ink)
            else:
                page_cache[p_num] = (None, None)
        return page_cache[p_num]

    answers, diagnostics = {}, {}
    for q in range(1, count + 1):
        idx = q - 1
        page_offset = idx // q_per_page
        q_in_page = idx % q_per_page
        col = q_in_page // qpc
        row = q_in_page % qpc

        target_p = base_page + page_offset
        rgb, is_ink = get_page_data(target_p)
        if is_ink is None:
            diagnostics[q] = {
                "method": "visual",
                "confidence": 0.0,
                "ratios": {opt: 0.0 for opt in options},
                "status": f"page_{target_p}_missing"
            }
            continue

        h, w = is_ink.shape
        cx0 = int((cfg["x_start"] + col * cfg["x_step"] * (len(options) + 1)) * w)
        cy = int((cfg["y_start"] + row * cfg["y_step"]) * h)
        ratios = {}
        for oi, option in enumerate(options):
            cx = cx0 + int(oi * cfg["option_step"] * w)
            r = max(5, int(cfg.get("bubble_radius", 10)))
            x1, x2 = max(0, cx-r), min(w, cx+r+1)
            y1, y2 = max(0, cy-r), min(h, cy+r+1)
            roi_ink = is_ink[y1:y2, x1:x2]
            if roi_ink.size == 0:
                ratios[option] = 0.0
                continue
            # Sample central disk inside the bubble
            yy, xx = np.ogrid[:roi_ink.shape[0], :roi_ink.shape[1]]
            mask = (xx - (roi_ink.shape[1]-1)/2)**2 + (yy - (roi_ink.shape[0]-1)/2)**2 <= max(1, (r * 0.72)**2)
            vals = roi_ink[mask]
            ratios[option] = float(np.mean(vals)) if vals.size else 0.0

        ranked = sorted(ratios.items(), key=lambda kv: kv[1], reverse=True)
        top_opt, top_ratio = ranked[0]
        second_ratio = ranked[1][1] if len(ranked) > 1 else 0
        marked = [opt for opt, ratio in ratios.items() if ratio >= 0.35]

        # Robust detection: filled bubble clearly exceeds background threshold
        if top_ratio >= 0.45 or (top_ratio >= 0.28 and top_ratio - second_ratio >= 0.12):
            if len(marked) == 1 or (len(marked) > 1 and top_ratio - second_ratio >= 0.15):
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

def extract_student_info(pdf_path, original_filename=""):
    """Attempt to extract candidate name and roll number from interactive form fields,
    roll number bubble widgets, or filename patterns."""
    name, roll_no = "", ""
    doc = None
    try:
        doc = fitz.open(pdf_path)
        roll_digits = {}
        for page in doc:
            for w in page.widgets() or []:
                fname = (w.field_name or "").strip()
                fval = str(w.field_value or "").strip()
                if not fval or fval.lower() in ("off", "false", "none", "0"):
                    continue
                if "candidatename" in fname.lower() or "candidate_name" in fname.lower():
                    if not name:
                        name = fval
                elif "rollnotext" in fname.lower() or "roll_no" in fname.lower():
                    if not roll_no:
                        roll_no = fval
                elif "rolldigit" in fname.lower():
                    m_chk = re.search(r"rolldigit_p\d+_(\d+)_(\d+)", fname, re.I)
                    if m_chk:
                        d_col = int(m_chk.group(1))
                        digit = m_chk.group(2)
                        roll_digits[d_col] = digit
                    else:
                        m_txt = re.search(r"rolldigit_p\d+_(\d+)$", fname, re.I)
                        if m_txt and fval:
                            d_col = int(m_txt.group(1))
                            roll_digits[d_col] = fval
        if not roll_no and roll_digits:
            roll_no = "".join([roll_digits[k] for k in sorted(roll_digits.keys())])
    except Exception:
        pass
    finally:
        if doc:
            try:
                doc.close()
            except Exception:
                pass

    # Visual roll number bubble detection if not found in AcroForm widgets:
    if not roll_no:
        try:
            rgb_p1 = render_pdf_page_rgb(pdf_path, 1)
            visual_roll = scan_roll_number_visual(rgb_p1)
            if visual_roll:
                roll_no = visual_roll
        except Exception:
            pass

    clean_base = os.path.splitext(os.path.basename(original_filename or ""))[0]
    if not roll_no and clean_base:
        m_roll = re.search(r"(?:roll|ht|hall|id|reg)?[ _\-#]*(\d{4,12})", clean_base, re.I)
        if m_roll:
            roll_no = m_roll.group(1)

    if not name and clean_base:
        cand_str = re.sub(r"^(?:sample|test|omr|exam)[ _\-]*", "", clean_base, flags=re.I)
        cand_str = re.sub(r"[ _\-]*(?:sheet|answersheet|response|omr)$", "", cand_str, flags=re.I)
        cand_str = cand_str.replace("_", " ").strip()
        if cand_str and not cand_str.isdigit():
            name = cand_str.title()
        elif roll_no:
            name = f"Candidate #{roll_no}"
        else:
            name = clean_base

    return name.strip(), roll_no.strip()

def annotate_evaluated_omr(source_pdf_bytes, cfg, details, score_summary, student_info=None):
    """Annotate evaluated OMR PDF with visual indicators and top scorecard stamp:
    - Official Evaluation Card at top right.
    - Vector green checkmark and green ring around correct bubbles.
    - Vector red cross 'X' and red circle around wrong bubbles.
    - Clear green ring and center dot indicating the correct answer on wrong questions.
    - Amber ring and dash '—' on unanswered questions.
    - Multi-page pagination aware.
    """
    doc = fitz.open(stream=source_pdf_bytes, filetype="pdf")

    # Auto-calibrate if old/custom template coordinates deviate from standard pink sheet
    if cfg.get("y_start", 0) < 0.33 or cfg.get("y_start", 0) > 0.45:
        cfg["y_start"] = 0.3658
        cfg["x_start"] = 0.1176
        cfg["x_step"] = 0.0433
        cfg["y_step"] = 0.0226
        cfg["option_step"] = 0.0311

    options = cfg.get("options", ["A", "B", "C", "D"])
    count = cfg.get("question_count", len(details))
    qpc = max(1, cfg.get("questions_per_column", 25))
    max_cols = max(1, cfg.get("max_cols_per_page", 4))
    q_per_page = max_cols * qpc

    GREEN = (0.09, 0.64, 0.29)
    GREEN_BG = (0.91, 0.98, 0.93)
    RED = (0.86, 0.15, 0.15)
    RED_BG = (0.99, 0.93, 0.93)
    AMBER = (0.85, 0.53, 0.04)
    GRAY_TEXT = (0.40, 0.43, 0.50)
    NAVY = (0.12, 0.16, 0.23)

    # 1. Top Evaluation Stamp Card on Page 1
    if len(doc) > 0:
        p1 = doc[0]
        pw, ph = p1.rect.width, p1.rect.height

        stamp_x0 = pw - 210.0
        stamp_y0 = 34.0
        stamp_w = 170.0
        stamp_h = 44.0
        stamp_rect = fitz.Rect(stamp_x0, stamp_y0, stamp_x0 + stamp_w, stamp_y0 + stamp_h)

        pct = score_summary.get("pct", 0.0)
        is_pass = pct >= 40.0
        theme_color = GREEN if is_pass else RED
        theme_bg = GREEN_BG if is_pass else RED_BG

        p1.draw_rect(stamp_rect, color=theme_color, fill=theme_bg, width=1.2)
        p1.insert_text(fitz.Point(stamp_x0 + 8, stamp_y0 + 13), "OFFICIAL EVALUATION", fontsize=6.8, fontname="hebo", color=theme_color)

        if student_info and student_info.get("roll_number"):
            roll_disp = f"ROLL: {student_info['roll_number']}"
            p1.insert_text(fitz.Point(stamp_x0 + stamp_w - 8 - len(roll_disp) * 4.3, stamp_y0 + 13), roll_disp, fontsize=6.8, fontname="hebo", color=NAVY)

        score_val = score_summary.get('score', 0)
        disp_score = int(score_val) if int(score_val) == score_val else round(score_val, 1)
        score_str = f"{disp_score} / {score_summary.get('total', count)}"
        p1.insert_text(fitz.Point(stamp_x0 + 8, stamp_y0 + 29), score_str, fontsize=13.0, fontname="hebo", color=theme_color)
        p1.insert_text(fitz.Point(stamp_x0 + 95, stamp_y0 + 28), f"({pct:.1f}%)", fontsize=10.0, fontname="hebo", color=theme_color)

        corr = score_summary.get("correct", 0)
        inc = score_summary.get("incorrect", 0)
        unans = score_summary.get("unanswered", 0)
        stat_line = f"Correct: {corr}  |  Wrong: {inc}  |  Blank: {unans}"
        p1.insert_text(fitz.Point(stamp_x0 + 8, stamp_y0 + 40), stat_line, fontsize=7.0, fontname="helv", color=NAVY)

    # 2. Annotate each question
    details_map = {d["question"]: d for d in details}
    bub_r_pt = 5.2

    for q in range(1, count + 1):
        d = details_map.get(q)
        if not d:
            continue

        page_idx = (q - 1) // q_per_page
        if page_idx >= len(doc):
            continue

        page = doc[page_idx]
        pw, ph = page.rect.width, page.rect.height

        q_in_page = (q - 1) % q_per_page
        col = q_in_page // qpc
        row = q_in_page % qpc

        cx0 = (cfg["x_start"] + col * cfg["x_step"] * (len(options) + 1)) * pw
        cy = (cfg["y_start"] + row * cfg["y_step"]) * ph
        opt_step = cfg["option_step"] * pw

        marked = (d.get("marked") or "").strip().upper()
        expected = (d.get("correct") or "").strip().upper()
        status = d.get("status", "")

        ix = cx0 - bub_r_pt - 21.0

        if status == "Correct" and marked in options:
            oi = options.index(marked)
            cx = cx0 + oi * opt_step
            page.draw_circle(fitz.Point(cx, cy), bub_r_pt + 2.0, color=GREEN, width=1.5)
            page.draw_line(fitz.Point(ix - 3.5, cy - 0.5), fitz.Point(ix - 1.0, cy + 2.8), color=GREEN, width=1.4)
            page.draw_line(fitz.Point(ix - 1.0, cy + 2.8), fitz.Point(ix + 3.8, cy - 3.5), color=GREEN, width=1.4)

        elif status == "Incorrect":
            if marked in options:
                oi = options.index(marked)
                cx = cx0 + oi * opt_step
                page.draw_circle(fitz.Point(cx, cy), bub_r_pt + 2.0, color=RED, width=1.5)
                cr = bub_r_pt + 0.5
                page.draw_line(fitz.Point(cx - cr, cy - cr), fitz.Point(cx + cr, cy + cr), color=RED, width=1.2)
                page.draw_line(fitz.Point(cx - cr, cy + cr), fitz.Point(cx + cr, cy - cr), color=RED, width=1.2)

            if expected in options:
                coi = options.index(expected)
                ccx = cx0 + coi * opt_step
                page.draw_circle(fitz.Point(ccx, cy), bub_r_pt + 2.0, color=GREEN, width=1.4)
                page.draw_circle(fitz.Point(ccx, cy), 1.8, color=GREEN, fill=GREEN)

            page.draw_line(fitz.Point(ix - 3.0, cy - 3.0), fitz.Point(ix + 3.0, cy + 3.0), color=RED, width=1.3)
            page.draw_line(fitz.Point(ix - 3.0, cy + 3.0), fitz.Point(ix + 3.0, cy - 3.0), color=RED, width=1.3)

        else: # Unanswered / blank
            if expected in options:
                coi = options.index(expected)
                ccx = cx0 + coi * opt_step
                page.draw_circle(fitz.Point(ccx, cy), bub_r_pt + 1.8, color=AMBER, width=1.1)
            page.draw_line(fitz.Point(ix - 3.0, cy), fitz.Point(ix + 3.0, cy), color=GRAY_TEXT, width=1.2)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def get_evaluated_pdf_for_result(result, upload_dir):
    """Retrieve or re-render source PDF, annotate evaluation results, and return PDF bytes."""
    payload = json.loads(result["details_json"])
    details = payload.get("details", [])

    cfg = None
    template_name = payload.get("template", "OMR Template")
    if result.get("template_id"):
        with db() as con:
            t = con.execute("SELECT * FROM templates WHERE id=?", (result["template_id"],)).fetchone()
            if t:
                try:
                    cfg = json.loads(t["config_json"])
                    template_name = t["name"]
                except Exception:
                    pass
    if not cfg:
        count = result["question_count"]
        cfg = {
            "question_count": count,
            "options": ["A", "B", "C", "D"],
            "page_number": 1,
            "layout_mode": "grid",
            "x_start": 0.1176, "y_start": 0.3658,
            "x_step": 0.0433, "y_step": 0.0226,
            "questions_per_column": 25, "option_step": 0.0311,
            "bubble_radius": 10, "max_cols_per_page": 4,
            "darkness_threshold": 90, "min_fill_ratio": 0.18, "max_fill_ratio": 0.70
        }

    stored_name = result.get("stored_filename") or result.get("filename")
    source_bytes = None
    if stored_name:
        fpath = os.path.join(upload_dir, stored_name)
        if os.path.exists(fpath):
            try:
                with open(fpath, "rb") as f:
                    source_bytes = f.read()
            except Exception:
                pass
        if not source_bytes:
            source_bytes = s3_download_file(f"omr-uploads/{stored_name}")

    if not source_bytes:
        doc = generate_omr_pdf(template_name, cfg, interactive=False)
        source_bytes = doc.tobytes()
        doc.close()

    pct = round((result["correct_count"] / result["question_count"] * 100.0), 1) if result["question_count"] else 0.0
    score_summary = {
        "score": result["score"],
        "total": result["question_count"],
        "correct": result["correct_count"],
        "incorrect": result["incorrect_count"],
        "unanswered": result["unanswered_count"],
        "pct": pct
    }
    student_info = {
        "name": result.get("candidate_name") or "",
        "roll_number": result.get("roll_number") or ""
    }

    eval_bytes = annotate_evaluated_omr(source_bytes, cfg, details, score_summary, student_info)
    if stored_name:
        s3_upload_file(eval_bytes, f"omr-evaluated/{stored_name}")
    return eval_bytes

def generate_batch_excel_workbook(results_list, test_name="OMR Evaluation", template_name="Standard Template"):
    """Generate a multi-sheet formatted Excel workbook (.xlsx) with:
    1. 'Results Summary' sheet: Rank, Roll Number, Student Name, Score, Percentage, Correct, Incorrect, Blank, Status.
    2. 'Question Matrix' sheet: Student answers question-by-question with green/red conditional fills.
    """
    wb = openpyxl.Workbook()

    ws1 = wb.active
    ws1.title = "Results Summary"
    ws1.views.sheetView[0].showGridLines = True

    navy_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    indigo_fill = PatternFill(start_color="312E81", end_color="312E81", fill_type="solid")
    pass_fill = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
    fail_fill = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
    gray_sub_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")

    white_bold = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Calibri", size=16, bold=True, color="0F172A")
    subtitle_font = Font(name="Calibri", size=10, italic=True, color="64748B")
    bold_font = Font(name="Calibri", size=11, bold=True)
    regular_font = Font(name="Calibri", size=10)
    pass_font = Font(name="Calibri", size=10, bold=True, color="166534")
    fail_font = Font(name="Calibri", size=10, bold=True, color="991B1B")

    thin_border = Border(
        left=Side(style='thin', color='CBD5E1'),
        right=Side(style='thin', color='CBD5E1'),
        top=Side(style='thin', color='CBD5E1'),
        bottom=Side(style='thin', color='CBD5E1')
    )

    ws1["A1"] = f"{test_name} — Evaluation Report"
    ws1["A1"].font = title_font
    ws1["A2"] = f"Template: {template_name}   |   Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}   |   Total Sheets: {len(results_list)}"
    ws1["A2"].font = subtitle_font

    headers = [
        "Rank", "Roll No.", "Student / Filename",
        "Score", "Max Marks", "Percentage",
        "Correct (✓)", "Incorrect (✗)", "Blank (—)",
        "Result Status", "Evaluated Date"
    ]
    header_row = 4
    ws1.row_dimensions[header_row].height = 26
    for c_idx, h in enumerate(headers, 1):
        cell = ws1.cell(row=header_row, column=c_idx, value=h)
        cell.fill = navy_fill
        cell.font = white_bold
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    sorted_results = sorted(results_list, key=lambda r: float(r["score"]), reverse=True)

    row_start = 5
    for rank, r in enumerate(sorted_results, 1):
        r_num = row_start + rank - 1
        ws1.row_dimensions[r_num].height = 20

        q_count = r["question_count"] or 1
        score_val = r["score"]
        pct = (score_val / q_count) * 100.0 if q_count else 0.0
        status_str = "PASSED" if pct >= 40.0 else "NEEDS REVIEW"

        display_name = r.get("candidate_name") or r.get("filename") or f"Student #{rank}"
        roll_val = r.get("roll_number") or "—"
        created_str = (r.get("created_at") or "")[:19].replace("T", " ")

        row_values = [
            rank,
            roll_val,
            display_name,
            score_val,
            q_count,
            pct / 100.0,
            r["correct_count"],
            r["incorrect_count"],
            r["unanswered_count"],
            status_str,
            created_str
        ]

        for c_idx, val in enumerate(row_values, 1):
            cell = ws1.cell(row=r_num, column=c_idx, value=val)
            cell.font = regular_font
            cell.border = thin_border

            if c_idx in (1, 2, 4, 5, 7, 8, 9, 11):
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif c_idx == 3:
                cell.alignment = Alignment(horizontal="left", vertical="center")
            elif c_idx == 6:
                cell.number_format = "0.0%"
                cell.alignment = Alignment(horizontal="right", vertical="center")
            elif c_idx == 10:
                cell.alignment = Alignment(horizontal="center", vertical="center")
                if status_str == "PASSED":
                    cell.fill = pass_fill
                    cell.font = pass_font
                else:
                    cell.fill = fail_fill
                    cell.font = fail_font

    if sorted_results:
        sum_row = row_start + len(sorted_results)
        ws1.row_dimensions[sum_row].height = 22
        ws1.cell(row=sum_row, column=1, value="").border = thin_border
        ws1.cell(row=sum_row, column=2, value="").border = thin_border
        cell_lbl = ws1.cell(row=sum_row, column=3, value="Class Average / Total")
        cell_lbl.font = bold_font
        cell_lbl.fill = gray_sub_fill
        cell_lbl.alignment = Alignment(horizontal="right", vertical="center")
        cell_lbl.border = thin_border

        avg_score = sum(r["score"] for r in sorted_results) / len(sorted_results)
        cell_avg = ws1.cell(row=sum_row, column=4, value=round(avg_score, 2))
        cell_avg.font = bold_font
        cell_avg.fill = gray_sub_fill
        cell_avg.alignment = Alignment(horizontal="center", vertical="center")
        cell_avg.border = thin_border

        max_q = sorted_results[0]["question_count"]
        cell_max = ws1.cell(row=sum_row, column=5, value=max_q)
        cell_max.font = bold_font
        cell_max.fill = gray_sub_fill
        cell_max.alignment = Alignment(horizontal="center", vertical="center")
        cell_max.border = thin_border

        avg_pct = (avg_score / max_q) if max_q else 0.0
        cell_avg_pct = ws1.cell(row=sum_row, column=6, value=avg_pct)
        cell_avg_pct.font = bold_font
        cell_avg_pct.fill = gray_sub_fill
        cell_avg_pct.number_format = "0.0%"
        cell_avg_pct.alignment = Alignment(horizontal="right", vertical="center")
        cell_avg_pct.border = thin_border

        for c in range(7, 12):
            c_cell = ws1.cell(row=sum_row, column=c, value="")
            c_cell.fill = gray_sub_fill
            c_cell.border = thin_border

    for col in ws1.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws1.column_dimensions[col_letter].width = max(max_len + 4, 11)

    # Sheet 2: Item Analysis Matrix
    if sorted_results and sorted_results[0].get("details_json"):
        try:
            first_payload = json.loads(sorted_results[0]["details_json"])
            q_list = [d["question"] for d in first_payload.get("details", [])]
            key_map = {d["question"]: d.get("correct", "") for d in first_payload.get("details", [])}

            if q_list:
                ws2 = wb.create_sheet(title="Item Analysis Matrix")
                ws2.views.sheetView[0].showGridLines = True

                ws2["A1"] = f"{test_name} — Question-by-Question Response Matrix"
                ws2["A1"].font = title_font
                ws2["A2"] = "Green = Correct answer | Red = Incorrect answer | Gray = Blank"
                ws2["A2"].font = subtitle_font

                h2 = ["Roll No.", "Candidate Name", "Total Score"] + [f"Q{q}" for q in q_list]
                ws2.row_dimensions[4].height = 24
                for c_idx, h in enumerate(h2, 1):
                    cell = ws2.cell(row=4, column=c_idx, value=h)
                    cell.fill = indigo_fill
                    cell.font = white_bold
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border = thin_border

                ws2.row_dimensions[5].height = 20
                ws2.cell(row=5, column=1, value="KEY").font = bold_font
                ws2.cell(row=5, column=2, value="Official Answer Key").font = bold_font
                ws2.cell(row=5, column=3, value=f"{len(q_list)} Qs").font = bold_font
                for c_idx in (1, 2, 3):
                    ws2.cell(row=5, column=c_idx).fill = gray_sub_fill
                    ws2.cell(row=5, column=c_idx).border = thin_border
                    ws2.cell(row=5, column=c_idx).alignment = Alignment(horizontal="center", vertical="center")
                for qi, q in enumerate(q_list, 1):
                    k_cell = ws2.cell(row=5, column=3 + qi, value=key_map.get(q, ""))
                    k_cell.font = bold_font
                    k_cell.fill = gray_sub_fill
                    k_cell.alignment = Alignment(horizontal="center", vertical="center")
                    k_cell.border = thin_border

                for s_idx, r in enumerate(sorted_results, 1):
                    r_num = 5 + s_idx
                    ws2.row_dimensions[r_num].height = 19
                    ws2.cell(row=r_num, column=1, value=r.get("roll_number") or "—").border = thin_border
                    ws2.cell(row=r_num, column=2, value=r.get("candidate_name") or r.get("filename")).border = thin_border
                    ws2.cell(row=r_num, column=3, value=r["score"]).border = thin_border
                    for c in (1, 3):
                        ws2.cell(row=r_num, column=c).alignment = Alignment(horizontal="center", vertical="center")

                    payload = json.loads(r["details_json"])
                    s_details = {d["question"]: d for d in payload.get("details", [])}

                    for qi, q in enumerate(q_list, 1):
                        qd = s_details.get(q, {})
                        marked = qd.get("marked", "")
                        status = qd.get("status", "")
                        qcell = ws2.cell(row=r_num, column=3 + qi, value=marked or "—")
                        qcell.alignment = Alignment(horizontal="center", vertical="center")
                        qcell.font = regular_font
                        qcell.border = thin_border

                        if status == "Correct":
                            qcell.fill = pass_fill
                            qcell.font = pass_font
                        elif status == "Incorrect":
                            qcell.fill = fail_fill
                            qcell.font = fail_font
                        else:
                            qcell.font = Font(name="Calibri", size=9, color="94A3B8")

                ws2.column_dimensions["A"].width = 14
                ws2.column_dimensions["B"].width = 24
                ws2.column_dimensions["C"].width = 13
                for qi in range(1, len(q_list) + 1):
                    col_ltr = get_column_letter(3 + qi)
                    ws2.column_dimensions[col_ltr].width = 6
        except Exception:
            pass

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

def generate_single_result_excel(result, details, template_name="Standard Template"):
    """Generate a clean student scorecard Excel workbook."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Student Scorecard"
    ws.views.sheetView[0].showGridLines = True

    navy_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    pass_fill = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
    fail_fill = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
    gray_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")

    white_bold = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Calibri", size=16, bold=True, color="0F172A")
    bold_font = Font(name="Calibri", size=11, bold=True)
    regular_font = Font(name="Calibri", size=10)
    pass_font = Font(name="Calibri", size=10, bold=True, color="166534")
    fail_font = Font(name="Calibri", size=10, bold=True, color="991B1B")

    thin_border = Border(
        left=Side(style='thin', color='CBD5E1'),
        right=Side(style='thin', color='CBD5E1'),
        top=Side(style='thin', color='CBD5E1'),
        bottom=Side(style='thin', color='CBD5E1')
    )

    ws["A1"] = f"{result['test_name']} — Candidate Scorecard"
    ws["A1"].font = title_font

    q_count = result["question_count"] or 1
    score = result["score"]
    pct = round((score / q_count) * 100.0, 1)

    info_rows = [
        ("Candidate Name", result.get("candidate_name") or result.get("filename") or "Candidate"),
        ("Roll Number", result.get("roll_number") or "—"),
        ("OMR Template", template_name),
        ("Evaluation Date", (result.get("created_at") or "")[:19].replace("T", " ")),
        ("Total Score", f"{score} / {q_count} ({pct}%)"),
        ("Breakdown", f"Correct: {result['correct_count']}  |  Incorrect: {result['incorrect_count']}  |  Blank: {result['unanswered_count']}")
    ]

    for i, (k, v) in enumerate(info_rows, 3):
        ws.cell(row=i, column=1, value=k).font = bold_font
        ws.cell(row=i, column=1).fill = gray_fill
        ws.cell(row=i, column=1).border = thin_border
        cell_v = ws.cell(row=i, column=2, value=v)
        cell_v.font = regular_font
        cell_v.border = thin_border
        if k == "Total Score":
            cell_v.font = pass_font if pct >= 40 else fail_font
            cell_v.fill = pass_fill if pct >= 40 else fail_fill

    tbl_start = len(info_rows) + 5
    headers = ["Question #", "Candidate Answer", "Answer Key", "Status", "Confidence", "Points"]
    ws.row_dimensions[tbl_start].height = 24
    for c_idx, h in enumerate(headers, 1):
        cell = ws.cell(row=tbl_start, column=c_idx, value=h)
        cell.fill = navy_fill
        cell.font = white_bold
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    for idx, d in enumerate(details, 1):
        r_num = tbl_start + idx
        ws.row_dimensions[r_num].height = 19
        diag = d.get("diagnostics", {})
        st = d.get("status", "")
        pts = 1 if st == "Correct" else 0

        row_vals = [
            d["question"],
            d["marked"] or "—",
            d["correct"],
            st,
            diag.get("confidence", "—"),
            pts
        ]

        for c_idx, val in enumerate(row_vals, 1):
            cell = ws.cell(row=r_num, column=c_idx, value=val)
            cell.font = regular_font
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center", vertical="center")

            if c_idx == 4:
                if st == "Correct":
                    cell.fill = pass_fill
                    cell.font = pass_font
                elif st == "Incorrect":
                    cell.fill = fail_fill
                    cell.font = fail_font

    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 4, 14)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

def generate_zip_evaluated_pdfs(results_list, app_upload_dir):
    """Bundle evaluated PDFs for all students in results_list into a single in-memory ZIP."""
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results_list:
            pdf_bytes = get_evaluated_pdf_for_result(r, app_upload_dir)
            if pdf_bytes:
                roll = (r.get("roll_number") or "").strip()
                name = (r.get("candidate_name") or r.get("filename") or f"student_{r['id']}").strip()
                safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", f"{roll}_{name}".strip("_"))
                zip_filename = f"{safe_name}_evaluated.pdf"
                zf.writestr(zip_filename, pdf_bytes)
    zip_buf.seek(0)
    return zip_buf.getvalue()

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
            sync_db_to_s3()
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
            sync_db_to_s3()
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
    sync_db_to_s3()
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

        uploads = request.files.getlist("pdf")
        valid_uploads = [u for u in uploads if u and u.filename and allowed_file(u.filename)]
        if not valid_uploads:
            flash("Please upload at least one valid PDF file.", "error")
            return render_template("scan.html", templates=templates, selected_template_id=selected_template_id)

        try:
            cfg = json.loads(template["config_json"])
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
            batch_id = secrets.token_hex(8)
            test_name = request.form.get("test_name", "").strip() or "OMR Test"
            created_ids = []
            errors = []

            for upload in valid_uploads:
                orig_filename = secure_filename(upload.filename)
                unique_name = f"{secrets.token_hex(8)}_{orig_filename}"
                path = os.path.join(UPLOAD_DIR, unique_name)
                upload.save(path)
                s3_upload_file(path, f"omr-uploads/{unique_name}")

                try:
                    doc = fitz.open(path)
                    if len(doc) > 100:
                        doc.close()
                        raise ValueError("PDF has too many pages (maximum 100).")
                    doc.close()

                    cand_name, roll_no = extract_student_info(path, orig_filename)
                    detected, diagnostics, method = scan_answers(path, cfg)

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
                        details.append({
                            "question": q,
                            "marked": given or "",
                            "correct": expected,
                            "status": status,
                            "diagnostics": diagnostics.get(q, {})
                        })

                    score = float(correct)
                    with db() as con:
                        cur = con.execute("""
                            INSERT INTO results(
                                user_id, template_id, test_name, filename, question_count,
                                correct_count, incorrect_count, unanswered_count, score,
                                details_json, created_at, batch_id, roll_number, candidate_name, stored_filename
                            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """, (
                            uid, template_id, test_name, orig_filename, cfg["question_count"],
                            correct, incorrect, unanswered, score,
                            json.dumps({
                                "details": details,
                                "method": method,
                                "template": template["name"],
                                "key": key_tokens
                            }),
                            datetime.utcnow().isoformat(timespec="seconds"),
                            batch_id, roll_no, cand_name, unique_name
                        ))
                        created_ids.append(cur.lastrowid)

                    sync_db_to_s3()
                    sync_result_to_firestore({
                        "test_name": test_name,
                        "candidate_name": cand_name,
                        "roll_number": roll_no,
                        "score": score,
                        "question_count": cfg["question_count"],
                        "correct_count": correct,
                        "incorrect_count": incorrect,
                        "unanswered_count": unanswered
                    })
                except Exception as file_err:
                    errors.append(f"{orig_filename}: {file_err}")

            if not created_ids:
                flash(f"Could not score uploaded files: {'; '.join(errors)}", "error")
                return render_template("scan.html", templates=templates, selected_template_id=selected_template_id)

            if errors:
                flash(f"Scored {len(created_ids)} file(s). Notice on failed files: {'; '.join(errors)}", "warning")
            else:
                flash(f"Successfully evaluated {len(created_ids)} answer sheet(s)!", "success")

            if len(created_ids) > 1:
                return redirect(url_for("batch_results", batch_id=batch_id))
            else:
                return redirect(url_for("result_detail", result_id=created_ids[0]))

        except Exception as e:
            flash(f"Evaluation error: {e}", "error")

    return render_template("scan.html", templates=templates, selected_template_id=selected_template_id)

@app.route("/batch/<batch_id>")
@login_required
def batch_results(batch_id):
    uid = session["user_id"]
    with db() as con:
        results = [dict(r) for r in con.execute(
            "SELECT * FROM results WHERE batch_id=? AND user_id=? ORDER BY score DESC, id ASC",
            (batch_id, uid)
        ).fetchall()]
    if not results:
        abort(404)

    test_name = results[0]["test_name"]
    template_name = "OMR Template"
    if results[0].get("template_id"):
        with db() as con:
            t = con.execute("SELECT name FROM templates WHERE id=?", (results[0]["template_id"],)).fetchone()
            if t:
                template_name = t["name"]

    total_students = len(results)
    total_score = sum(r["score"] for r in results)
    avg_score = round(total_score / total_students, 1) if total_students else 0
    highest_score = max(r["score"] for r in results) if results else 0
    lowest_score = min(r["score"] for r in results) if results else 0
    q_count = results[0]["question_count"] or 1
    avg_pct = round((avg_score / q_count) * 100, 1) if q_count else 0
    pass_count = sum(1 for r in results if (r["score"] / q_count * 100) >= 40.0)
    pass_rate = round((pass_count / total_students) * 100, 1) if total_students else 0

    return render_template(
        "batch_results.html",
        batch_id=batch_id,
        results=results,
        test_name=test_name,
        template_name=template_name,
        total_students=total_students,
        avg_score=avg_score,
        avg_pct=avg_pct,
        highest_score=highest_score,
        lowest_score=lowest_score,
        pass_rate=pass_rate,
        pass_count=pass_count
    )

@app.route("/batch/<batch_id>/excel")
@login_required
def batch_excel(batch_id):
    uid = session["user_id"]
    with db() as con:
        results = [dict(r) for r in con.execute(
            "SELECT * FROM results WHERE batch_id=? AND user_id=? ORDER BY score DESC, id ASC",
            (batch_id, uid)
        ).fetchall()]
    if not results:
        abort(404)

    test_name = results[0]["test_name"]
    template_name = "OMR Template"
    if results[0].get("template_id"):
        with db() as con:
            t = con.execute("SELECT name FROM templates WHERE id=?", (results[0]["template_id"],)).fetchone()
            if t:
                template_name = t["name"]

    excel_bytes = generate_batch_excel_workbook(results, test_name, template_name)
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", test_name.strip()).strip("_")
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"{safe_name}_batch_results.xlsx"
    )

@app.route("/batch/<batch_id>/download-all-pdfs")
@login_required
def batch_all_pdfs(batch_id):
    uid = session["user_id"]
    with db() as con:
        results = [dict(r) for r in con.execute(
            "SELECT * FROM results WHERE batch_id=? AND user_id=?",
            (batch_id, uid)
        ).fetchall()]
    if not results:
        abort(404)

    test_name = results[0]["test_name"]
    zip_bytes = generate_zip_evaluated_pdfs(results, UPLOAD_DIR)
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", test_name.strip()).strip("_")
    return send_file(
        io.BytesIO(zip_bytes),
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{safe_name}_all_evaluated_pdfs.zip"
    )

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

@app.route("/results/<int:result_id>/evaluated-pdf")
@login_required
def result_evaluated_pdf(result_id):
    uid = session["user_id"]
    with db() as con:
        r = con.execute("SELECT * FROM results WHERE id=? AND user_id=?", (result_id, uid)).fetchone()
    if not r:
        abort(404)

    pdf_bytes = get_evaluated_pdf_for_result(dict(r), UPLOAD_DIR)
    roll = (r["roll_number"] or "").strip()
    name = (r["candidate_name"] or r["filename"] or f"student_{r['id']}").strip()
    safe_label = re.sub(r"[^a-zA-Z0-9_\-]+", "_", f"{roll}_{name}".strip("_"))
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{safe_label}_evaluated.pdf"
    )

@app.route("/results/<int:result_id>/excel")
@login_required
def result_excel(result_id):
    uid = session["user_id"]
    with db() as con:
        r = con.execute("SELECT * FROM results WHERE id=? AND user_id=?", (result_id, uid)).fetchone()
    if not r:
        abort(404)

    payload = json.loads(r["details_json"])
    template_name = payload.get("template", "OMR Template")
    excel_bytes = generate_single_result_excel(dict(r), payload.get("details", []), template_name)

    roll = (r["roll_number"] or "").strip()
    name = (r["candidate_name"] or r["filename"] or f"student_{r['id']}").strip()
    safe_label = re.sub(r"[^a-zA-Z0-9_\-]+", "_", f"{roll}_{name}".strip("_"))
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"{safe_label}_scorecard.xlsx"
    )

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

@app.route("/results/export-all-excel")
@login_required
def export_all_excel():
    uid = session["user_id"]
    with db() as con:
        results = [dict(r) for r in con.execute(
            "SELECT * FROM results WHERE user_id=? ORDER BY id DESC",
            (uid,)
        ).fetchall()]
    if not results:
        flash("No results to export yet.", "info")
        return redirect(url_for("dashboard"))

    excel_bytes = generate_batch_excel_workbook(results, "All Scored Tests", "All Templates")
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="all_omr_results.xlsx"
    )

@app.route("/results/<int:result_id>/delete", methods=["POST"])
@login_required
def delete_result(result_id):
    with db() as con:
        con.execute("DELETE FROM results WHERE id=? AND user_id=?", (result_id, session["user_id"]))
    sync_db_to_s3()
    flash("Result deleted.", "success")
    return redirect(url_for("dashboard"))

with app.app_context():
    init_db()

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
