# app.py (defensive, google-genai migration, with safe SHAP plotting + soft prompts)
import os
import sys
import logging
from flask import Flask, request, render_template
import numpy as np
import warnings
from joblib import load
from feature import FeatureExtraction
import json
import requests
# new SDK import (already in your env)
import google.genai as genai
import re
import shap

# IMPORTANT: force non-interactive backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from io import BytesIO
import base64
from urllib.parse import urlparse

# -------- Logging ----------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# -------- Load ML model ---------------------------------------------------
MODEL_PATH = "pickle/model-updated-final.pkl"
try:
    gbc = load(MODEL_PATH)
    logger.info("Loaded model from %s", MODEL_PATH)
except Exception as e:
    logger.exception("Failed loading model. Make sure file exists and is compatible:")
    raise

# -------- Configure Gemini (google-genai client, non-fatal) ----------------
genai_key = os.environ.get("GENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
LLM_ENABLED = bool(genai_key)
CLIENT = None

if LLM_ENABLED:
    try:
        CLIENT = genai.Client(api_key=genai_key)
        logger.info("google-genai Client created (LLM enabled).")
    except Exception:
        logger.exception("Failed to create google-genai Client. Disabling LLM.")
        LLM_ENABLED = False
        CLIENT = None
else:
    logger.warning("GENAI_API_KEY / GOOGLE_API_KEY not set; running in ML-only mode (LLM disabled).")


# ---------- NEW: Soft Prompt Utilities (Non-breaking Add-on) ---------------
def load_soft_prompt():
    """
    Loads a soft prompt prefix. 
    Helps guide the Gemma model toward consistent phishing analysis.
    """
    return (
        "[SOFT_PROMPT_PHISHGUARD] "
        "You are a phishing-detection expert. "
        "Analyze the webpage for brand impersonation, fake login pages, "
        "malicious redirects, suspicious forms, script injections, and obfuscated URLs. "
        "Return a structured JSON with: verdict, confidence, indicators, and target_brand."
    )


def apply_soft_prompt(html_content, soft_prompt):
    """
    Prepends soft prompt to HTML snippet — does not alter any core logic.
    """
    if not html_content:
        return soft_prompt
    return soft_prompt + "\n\n" + html_content


# -------- Helpers ---------------------------------------------------------
def normalize_url(u: str) -> str:
    """Ensure URL has a scheme; default to https:// if missing."""
    if not u:
        return u
    u = u.strip()
    parsed = urlparse(u)
    if parsed.scheme == "":
        return "https://" + u
    return u

def safe_llm_result():
    return {"verdict": "uncertain", "confidence": 0.0, "reasons": ["LLM disabled or error"], "target_brand": None}


# -------- LLM call (with soft prompts integrated safely) ------------------
def analyze_with_llm_gemma(url, html_content=""):
    try:
        endpoint = "https://teeny-owlishly-mauro.ngrok-free.dev/"

        # -------- NEW: Soft prompts applied here -----------------
        soft_prompt = load_soft_prompt()
        enhanced_html = apply_soft_prompt(html_content, soft_prompt)
        # ----------------------------------------------------------

        payload = {
            "url": url,
            "html_snippet": enhanced_html
        }

        headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true"
        }

        resp = requests.post(endpoint, json=payload, headers=headers, timeout=40)

        if resp.status_code != 200:
            return {
                "verdict": "error",
                "confidence": 0,
                "reasons": [f"Gemma server error {resp.status_code}", resp.text],
                "target_brand": None
            }

        data = resp.json()
        mapped = {
            "verdict": data.get("verdict", "unknown"),
            "confidence": data.get("confidence", 0),
            "reasons": data.get("indicators", []),
            "target_brand": data.get("target_brand", None),
            "explanation": data.get("explanation", "")
        }

        return mapped

    except Exception as e:
        return {
            "verdict": "error",
            "confidence": 0,
            "reasons": [f"Gemma server unreachable: {str(e)}"],
            "target_brand": None
        }



# -------- Flask app -------------------------------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            raw_url = request.form.get("url", "").strip()
            if not raw_url:
                return render_template("index1.html", error="Please enter a URL.", **empty_context())

            url = normalize_url(raw_url)
            logger.info("Analyzing URL: %s", url)

            # Feature extraction
            try:
                obj = FeatureExtraction(url)
                features = obj.getFeaturesList()
                x = np.array(features).reshape(1, -1)
            except Exception:
                logger.exception("Feature extraction failed for %s", url)
                return render_template("index1.html", error="Feature extraction failed. See logs.", **empty_context())

            # ML prediction
            try:
                y_pred = int(gbc.predict(x)[0])
                proba = gbc.predict_proba(x)[0].astype(float)
                y_pro_phishing = float(proba[0])
                y_pro_non_phishing = float(proba[1])
                ml_verdict = "LEGITIMATE" if y_pred == 1 else "PHISHING"
                ml_conf = max(y_pro_phishing, y_pro_non_phishing)
            except Exception:
                logger.exception("Model prediction failed:")
                return render_template("index1.html", error="Model prediction failed. Check model compatibility.", **empty_context())

            # Fetch HTML
            html_content = ""
            try:
                headers = {"User-Agent": "PhishGuardBot/1.0 (+https://example.com)"}
                resp = requests.get(url, timeout=8, headers=headers, allow_redirects=True)
                html_content = resp.text
            except Exception as e:
                logger.warning("Failed to fetch HTML for %s: %s", url, e)
                html_content = f"Error fetching page: {e}"

            # LLM + soft prompts
            llm_result = analyze_with_llm_gemma(url, html_content)
            llm_verdict = llm_result.get("verdict", "error")
            llm_confidence = float(llm_result.get("confidence", 0.0))
            llm_reasons = llm_result.get("reasons", [])
            target_brand = llm_result.get("target_brand", None)

            # Combined score
            ml_weight = 0.4
            llm_weight = 0.6
            ml_risk = 1 if ml_verdict == "PHISHING" else 0
            llm_risk = 1 if llm_verdict == "phishing" else 0
            combined_risk_score = round((ml_risk * ml_conf * ml_weight) + (llm_risk * llm_confidence * llm_weight), 2)

            if combined_risk_score >= 0.7:
                final_verdict = "PHISHING"
            elif combined_risk_score <= 0.3:
                final_verdict = "LEGITIMATE"
            else:
                final_verdict = "UNCERTAIN"

            # SHAP Circular Bar Graph
            ml_feature_plot = None
            try:
                explainer = shap.TreeExplainer(gbc)
                shap_values = explainer.shap_values(x)

                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_vals_for_phishing = shap_values[1]
                else:
                    shap_vals_for_phishing = shap_values

                abs_shap = np.abs(shap_vals_for_phishing[0])

                feature_names = [
                    "Using IP", "Long URL", "Short URL", "Symbol @", "Redirecting //", "Prefix-Suffix",
                    "Subdomains", "HTTPS", "Domain Reg Length", "Favicon",
                    "NonStd Port", "HTTPS in Domain", "Request URL", "Anchor URL", "Links in Script Tags",
                    "Server Form Handler", "Info Email", "Abnormal URL", "Website Forwarding", "Status Bar Custom",
                    "Disable Right Click", "Popup Window", "Iframe Redirection", "Age of Domain",
                    "DNS Recording", "Website Traffic", "Page Rank", "Google Index", "Links Pointing To Page",
                    "Stats Report"
                ]

                n_feats = abs_shap.shape[0]
                if len(feature_names) < n_feats:
                    feature_names += [f"feat_{i}" for i in range(len(feature_names), n_feats)]
                else:
                    feature_names = feature_names[:n_feats]

                top_idx = np.argsort(abs_shap)[-5:][::-1]
                top_features = [feature_names[i] for i in top_idx]
                top_values = [abs_shap[i] for i in top_idx]

                others_value = np.sum(abs_shap) - np.sum(top_values)
                top_features.append("Others")
                top_values.append(others_value)

                denom = np.sum(abs_shap)
                if denom == 0 or np.isnan(denom):
                    raise ValueError("SHAP total is zero; cannot plot")
                top_values_percent = np.array(top_values) / denom * 100

                theta = np.linspace(0.0, 2 * np.pi, len(top_values_percent), endpoint=False)
                radii = top_values_percent
                width = 2*np.pi/len(top_values_percent) * 0.5

                fig, ax = plt.subplots(subplot_kw={'polar': True}, figsize=(5,5))
                colors = plt.cm.plasma(np.linspace(0.3, 0.8, len(top_values_percent)))
                bars = ax.bar(theta, radii, width=width, bottom=0.0, color=colors, edgecolor='white', linewidth=1.2)

                for i, (bar, label, value) in enumerate(zip(bars, top_features, top_values_percent)):
                    angle = np.rad2deg(theta[i])
                    rotation = angle if angle <= 180 else angle-180
                    ha = 'left' if angle <= 180 else 'right'
                    ax.text(theta[i], radii[i]+2, f"{label}\n{value:.1f}%", ha=ha, va='center',
                            rotation=rotation, rotation_mode='anchor', fontsize=10, color='white', weight='bold')

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor('none')
                plt.tight_layout()

                buf = BytesIO()
                plt.savefig(buf, format="png", bbox_inches='tight', transparent=True)
                buf.seek(0)
                ml_feature_plot = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()

            except Exception as e:
                logger.exception("SHAP plotting failed: %s", e)
                ml_feature_plot = None

            return render_template(
                "index1.html",
                url=url,
                ml_verdict=ml_verdict,
                ml_conf=ml_conf,
                llm_verdict=llm_verdict,
                llm_conf=llm_confidence,
                llm_reasons=llm_reasons,
                target_brand=target_brand,
                combined_risk_score=combined_risk_score,
                final_verdict=final_verdict,
                ml_feature_plot=ml_feature_plot,
                error=None
            )

        # GET
        return render_template("index1.html", **empty_context())
    except Exception:
        logger.exception("Unhandled error in index handler")
        return render_template("index1.html", error="Internal server error. Check logs."), 500


def empty_context():
    return dict(url=None, ml_verdict=None, ml_conf=None, llm_verdict=None,
                llm_conf=None, llm_reasons=[], target_brand=None,
                combined_risk_score=None, final_verdict=None, ml_feature_plot=None)


if __name__ == "__main__":
    app.run(debug=True)
