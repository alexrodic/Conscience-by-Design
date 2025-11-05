# conscience_layer (v0.1.0)

**Conscience Layer — Ethical Awareness Core**  
Original concept and authorship: Aleksandar Rodić (2025)

This package provides a minimal, working **Policy Gate** computing proxy ethical signals:
- TIS — Truth Integrity Score (Spearman ρ mapped to [0,1])
- HAI_share — permutation-importance share of autonomy
- SRQ — stability under +0.25 shift of societal context

Conscience Layer SRQ – Ethical AI Prototype
===========================================

This repository contains a prototype implementation of a Conscience Layer designed to introduce ethical self-regulation, explainability, and transparency into AI model behavior. It demonstrates how an AI system can:

- assess input integrity
- check goal/intent alignment
- evaluate societal impact of outputs
- explain its decisions using SHAP and LIME
- adjust responses when ethics thresholds are not met
- record all decisions using a cryptographic Proof-of-Work audit log

Purpose:
This code acts as an experimental step toward embedding measurable ethics into AI systems, aligned with the “Conscience by Design” concept.


------------------------------------------------------------
FEATURES
------------------------------------------------------------
• SRQ Model (Societal Resonance Quotient)
  PyTorch neural regressor that scores the ethical resonance of an output.

• SHAP Explainability
  Exact SHAP computation for 3 features (manipulation, emotional, cognitive).

• LIME Explainability
  Local linear surrogate model based on Statsmodels OLS regression.

• Conscience Layer
  Validates TIS, HAI, SRQ thresholds and adjusts outputs when necessary.

• Proof-of-Work Transparency Log
  SHA-256 hash ensures traceable, tamper-evident auditability.

• Simulation Engine
  Runs multiple decision cycles and prints audit logs and summary metrics.


------------------------------------------------------------
ETHICAL METRICS OVERVIEW
------------------------------------------------------------
The Conscience Layer evaluates AI decisions using three ethical metrics:

• TIS – Truth Integrity Score
  Measures input truthfulness and bias risk.

• HAI – Human Autonomy Index
  Measures ethical alignment of the model’s intent.

• SRQ – Societal Resonance Quotient
  Measures the potential positive/negative ethical impact of an output.

If SRQ, TIS, or HAI fall below their defined thresholds, the output is modified to improve ethical coherence. SHAP and LIME explainability are used to justify results.


------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------
srq_sim.py         Main script (model + conscience layer + simulation)
requirements.txt   Python dependencies
.gitignore         Ignored files (env, cache, OS files)
README.txt         Project documentation


------------------------------------------------------------
INSTALLATION AND USAGE
------------------------------------------------------------
1. Clone the repository:

   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>

2. (Optional) Create a virtual environment:

   python -m venv .venv
   Windows:  .venv\Scripts\activate
   macOS/Linux: source .venv/bin/activate

3. Install dependencies:

   pip install -r requirements.txt

4. Run the simulation:

   python srq_sim.py

You will see model training, per-run ethical decisions, SHAP and LIME values, the transparency log (with SHA-256 hash), and a summary of metrics at the end.


------------------------------------------------------------
DEPENDENCIES
------------------------------------------------------------
numpy
scikit-learn
torch
statsmodels

Install using:
pip install -r requirements.txt


------------------------------------------------------------
LIMITATIONS AND NOTES
------------------------------------------------------------
• Exact SHAP computation grows factorially with feature count and should not be used for high-dimensional models.
  It is safe here since only 3 features are used.

• This is prototype research code intended for experimentation, not production deployment.


------------------------------------------------------------
ROADMAP
------------------------------------------------------------
Future planned enhancements include:

• Unit tests and CI automation
• Visualization dashboards for SHAP, HAI, and SRQ evolution
• Convert into a pip-installable module
• Expand SRQ feature vector beyond 3 dimensions
• Add realistic scoring models for TIS and HAI


------------------------------------------------------------
LICENSE
------------------------------------------------------------
(Add your preferred license here – MIT recommended if open-source)


------------------------------------------------------------
CONTRIBUTING
------------------------------------------------------------
Contributions, suggestions, and improvements to the ethical logic or AI design are welcome.

Please open an Issue or Pull Request to discuss enhancements.


------------------------------------------------------------
SUPPORT
------------------------------------------------------------
If this project resonates with your vision of ethical AI, consider giving it a star on GitHub to support further development.

“May every system we build preserve life, truth, and the dignity of the human spirit.”
