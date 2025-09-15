#importing required libraries
from flask import Flask, request, render_template
import numpy as np
import warnings
import pickle
from feature import FeatureExtraction
import json
import requests
import google.generativeai as genai
import re
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64

warnings.filterwarnings('ignore')

# =========================
# Load ML model
# =========================
file = open("pickle/model-updated.pkl", "rb")
gbc = pickle.load(file)
file.close()

# =========================
# Configure Gemini API
# =========================
import os
genai_key = os.environ.get("GENAI_API_KEY")
if not genai_key:
    raise RuntimeError("Set the GENAI_API_KEY environment variable")
genai.configure(api_key=genai_key)

# =========================
# Function to analyze with Gemini
# =========================
def analyze_with_llm_gemini(url, html_content=""):
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

    prompt = f"""
    **Role:** You are a senior cybersecurity analyst.

    **Task:** Analyze the provided URL and HTML content snippet for signs of a phishing website.

    **Output Format:** You MUST output a valid JSON object with the following structure and nothing else:
    {{
      "verdict": "phishing", // or "legitimate" or "uncertain"
      "confidence": 0.85, // a float between 0 and 1
      "reasons": ["Reason 1", "Reason 2", "Reason 3"],
      "target_brand": "PayPal" // or null if no clear brand is being impersonated
    }}

    **Data to Analyze:**
    - URL: {url}
    - HTML Content Snippet: {html_content[:1500]}...
    """
    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    # Extract JSON with regex
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception as e:
            return {
                "verdict": "error",
                "confidence": 0,
                "reasons": [f"JSON parse error: {e}", raw_text],
                "target_brand": None,
            }
    else:
        return {
            "verdict": "error",
            "confidence": 0,
            "reasons": ["No JSON found", raw_text],
            "target_brand": None,
        }

# =========================
# Flask App
# =========================
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, -1)

        # ML Prediction
        y_pred = gbc.predict(x)[0]
        y_pro_phishing = gbc.predict_proba(x)[0, 0]
        y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

        ml_verdict = "LEGITIMATE" if y_pred == 1 else "PHISHING"
        ml_conf = max(y_pro_phishing, y_pro_non_phishing)

        # Fetch HTML
        try:
            response = requests.get(url, timeout=5)
            html_content = response.text
        except Exception as e:
            html_content = f"Error fetching page: {e}"

        # Gemini LLM
        llm_result = analyze_with_llm_gemini(url, html_content)

        # =========================
        # Combined Phishing Risk
        # =========================
        ml_weight = 0.4
        llm_weight = 0.6

        ml_risk = 1 if ml_verdict == "PHISHING" else 0
        llm_risk = 1 if llm_result["verdict"] == "phishing" else 0

        combined_risk_score = (ml_risk * ml_conf * ml_weight) + (llm_risk * llm_result["confidence"] * llm_weight)
        combined_risk_score = round(combined_risk_score, 2)

        if combined_risk_score >= 0.7:
            final_verdict = "PHISHING"
        elif combined_risk_score <= 0.3:
            final_verdict = "LEGITIMATE"
        else:
            final_verdict = "UNCERTAIN"

        # =========================
        # SHAP Circular Bar Graph for top 5 features (Compact & Dark-friendly)
        # =========================
        explainer = shap.TreeExplainer(gbc)
        shap_values = explainer.shap_values(x)
        # Handle binary and multiclass output
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_vals_for_phishing = shap_values[1]  # class=1 (phishing)
        else:
            shap_vals_for_phishing = shap_values  # binary case

        abs_shap = np.abs(shap_vals_for_phishing[0])

        # Feature names
        feature_names = [
            "Using IP", "Long URL", "Short URL", "Symbol @", "Redirecting //", "Prefix-Suffix",
            "Subdomains", "HTTPS", "Domain Reg Length", "Favicon",
            "NonStd Port", "HTTPS in Domain", "Request URL", "Anchor URL", "Links in Script Tags",
            "Server Form Handler", "Info Email", "Abnormal URL", "Website Forwarding", "Status Bar Custom",
            "Disable Right Click", "Popup Window", "Iframe Redirection", "Age of Domain",
            "DNS Recording", "Website Traffic", "Page Rank", "Google Index", "Links Pointing To Page",
            "Stats Report"
        ]

        # Sort top 5
        top_idx = np.argsort(abs_shap)[-5:][::-1]
        top_features = [feature_names[i] for i in top_idx]
        top_values = [abs_shap[i] for i in top_idx]

        # Others
        others_value = np.sum(abs_shap) - np.sum(top_values)
        top_features.append("Others")
        top_values.append(others_value)

        # Convert to percentages
        top_values_percent = np.array(top_values) / np.sum(abs_shap) * 100

        # Circular bar plot
        theta = np.linspace(0.0, 2 * np.pi, len(top_values_percent), endpoint=False)
        radii = top_values_percent
        width = 2*np.pi/len(top_values_percent) * 0.5  # narrower bars for compactness

        fig, ax = plt.subplots(subplot_kw={'polar': True}, figsize=(5,5))  # smaller figure
        # Dark-friendly gradient colors
        colors = plt.cm.plasma(np.linspace(0.3, 0.8, len(top_values_percent)))
        bars = ax.bar(theta, radii, width=width, bottom=0.0, color=colors, edgecolor='white', linewidth=1.2)

        # Labels outside bars (adjusted for clarity)
        for i, (bar, label, value) in enumerate(zip(bars, top_features, top_values_percent)):
            angle = np.rad2deg(theta[i])
            rotation = angle if angle <= 180 else angle-180
            ha = 'left' if angle <= 180 else 'right'
            ax.text(theta[i], radii[i]+2, f"{label}\n{value:.1f}%", ha=ha, va='center', rotation=rotation,
                    rotation_mode='anchor', fontsize=10, color='white', weight='bold')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('none')  # transparent background for dark card
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', transparent=True)
        buf.seek(0)
        ml_feature_plot = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()


        # Render template
        return render_template(
            "index1.html",
            url=url,
            ml_verdict=ml_verdict,
            ml_conf=ml_conf,
            llm_verdict=llm_result["verdict"],
            llm_conf=llm_result["confidence"],
            llm_reasons=llm_result["reasons"],
            target_brand=llm_result["target_brand"],
            combined_risk_score=combined_risk_score,
            final_verdict=final_verdict,
            ml_feature_plot=ml_feature_plot
        )

    return render_template("index1.html", ml_verdict=None, llm_verdict=None,  ml_conf=None, llm_conf=None, llm_reasons=[], target_brand=None, combined_risk_score=None, final_verdict=None, ml_feature_plot=None)

if __name__ == "__main__":
    app.run(debug=True)
