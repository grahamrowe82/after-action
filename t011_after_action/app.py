from __future__ import annotations

from flask import Flask, render_template, request

try:
    from .logic import assemble
except ImportError:  # pragma: no cover - fallback when run as script
    from logic import assemble

app = Flask(__name__)

INPUT_LIMIT = 8000


@app.route("/", methods=["GET"])
def index():
    results = assemble("")
    return render_template(
        "index.html",
        results=results,
        text="",
        error=None,
    )


@app.route("/compress", methods=["POST"])
def compress():
    text = request.form.get("notes", "").strip()
    error = None
    if len(text) > INPUT_LIMIT:
        error = "Trim input to 8,000 characters."
        results = assemble("")
    else:
        results = assemble(text)
    return render_template(
        "index.html",
        results=results,
        text=text,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
