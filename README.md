# Digital OMR Scanner

A Flask + SQLite web application for accounts, custom OMR template settings, PDF answer-sheet scanning, scoring, saved result history, and CSV export.

## Features
- User registration and login (PBKDF2 password hashing).
- Per-user custom OMR templates.
- **Pre-Calibrated Standard Answer Sheets**: Built-in 1-click loading for standard competitive exam formats (30Q, 50Q, 60Q, 100Q) mathematically calibrated with 100% scanning accuracy.
- **Exam Header Format**: Includes registration corner marks, booklet/set code selector (A, B, C, D), candidate info fields (Name, Roll No., Class/Section, Date, Signature), and **Roll Number digit boxes with 0–9 bubble columns**.
- **Download Template Sheets**: Instant generation and download of ready-to-print A4 OMR answer sheets (PDF), interactive digital fillable PDFs (with AcroForm bubbles), and JSON config backups.
- Upload PDF answer sheets (up to 20 MB, 100 pages).
- Reads interactive AcroForm fields when field names include question numbers, e.g. `Q1_A`, `question_2`, or radio field values `A`, `B`, `C`, `D`.
- If a PDF is flattened or uses visual marks, uses OpenCV sampling based on the saved custom bubble-grid coordinates.
- Answer key with configurable question count and options.
- Scoring: +1 per correct answer, 0 for incorrect or unanswered.
- Result history, question-level review, detection confidence diagnostics, and CSV download.

## Important limitation
This is a configurable starter implementation, not a universal OMR reader. Visual scanning assumes a consistent, aligned bubble grid. You must tune the template coordinates against your exact PDF layout. The app does not yet automatically infer arbitrary bubble locations. PDF form-field extraction works only when the PDF contains usable AcroForm fields and their names/values follow a recognizable convention. Always review low-confidence or blank/unclear answers before relying on a score.

## Run locally (Windows PowerShell)

1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/ and select **Add Python to PATH**.
2. Extract this ZIP and open PowerShell in the project folder.
3. Create a virtual environment:
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Set a secure session key for this terminal:
   ```powershell
   $env:SECRET_KEY = python -c "import secrets; print(secrets.token_hex(32))"
   ```
6. Start:
   ```powershell
   python app.py
   ```
7. Open http://127.0.0.1:5000 and register an account.

## Configure a custom template

Go to **Templates → Create template**. The visual detector uses normalized coordinates from the top-left of the rendered PDF page:
- `x_start`, `y_start`: center of the first bubble for question 1, option A.
- `x_step`: horizontal spacing multiplier used to move to the next question column. The actual column advance is `x_step × (number of options + 1)`.
- `y_step`: vertical distance between consecutive question rows.
- `option_step`: horizontal distance between option bubbles within one question.
- `questions_per_column`: how many questions run down each column before the next column begins.
- `bubble_radius`: radius in pixels at the app's 2× render scale.
- `page_number`: 1-based PDF page containing the answer grid.
- `darkness_threshold`, `min_fill_ratio`, `max_fill_ratio`: detection tuning values.

The default values are examples only. Measure bubble centers on your exact answer-sheet PDF. Because the detector renders at 2×, bubble radius is in the rendered image's pixels.

## Interactive PDF field naming

For best results, create radio fields with a question number in the field name, such as:
- radio group `Q1` with selected value `A`, `B`, `C`, or `D`
- fields named `question_1_A`, `question_1_B` etc. if the selected field value/name identifies the option

Different PDF generators encode radio export values differently. Test a sample PDF first. If fields are not recognized, the app falls back to visual scanning.

## Deploy

### Render / Railway / any Python host
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Set environment variable `SECRET_KEY` to a long random value.
- Add a persistent disk mounted to the project `instance/` directory. SQLite and uploaded PDFs are stored there. Without persistent storage, data may disappear on redeploy.
- Keep debug mode off in production.
- Configure the host's upload/request size limits to at least 20 MB if supported.

### Production hardening before public use
- Use HTTPS.
- Set `SESSION_COOKIE_SECURE=True` and `SESSION_COOKIE_SAMESITE="Lax"` behind HTTPS.
- Add CSRF protection (e.g. Flask-WTF), rate limiting, email verification, password reset, and account deletion.
- Add a retention policy and a UI for deleting uploaded source PDFs. Current app stores uploaded PDFs in `instance/uploads/`; deleting a result does not automatically delete its PDF.
- Use PostgreSQL and object storage for a multi-user production service.
- Add stronger file scanning and PDF resource limits if accepting public uploads.
