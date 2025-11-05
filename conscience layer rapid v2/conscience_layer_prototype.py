# =============================================================================
# Conscience Layer — Ethical Awareness Core
# -----------------------------------------------------------------------------
# Original concept and authorship: Aleksandar Rodić (2025)
# Founder of the Conscience by Design Initiative
#
# Description:
# The Conscience Layer represents the ethical awareness core —
# a universal framework designed to embed conscience, moral reflection,
# and human-centered alignment directly into artificial intelligence systems.
#
# Purpose:
# To ensure that every intelligent system acts with awareness of truth,
# autonomy, and social resonance — transforming technology into a
# responsible extension of human conscience.
#
# Donation:
# Donated freely to humanity as the moral heart for future AI systems.
#
# License:
# Dual License —
#   • Text, framework, and documentation: Creative Commons Attribution 4.0 (CC BY 4.0)
#   • Source code and implementations: MIT License
#
# (c) 2025 Aleksandar Rodić — Conscience by Design
# "The true evolution of intelligence begins when technology learns to care."
# =============================================================================


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
import math
import itertools
import statsmodels.api as sm

# -----------------------------
# SRQ Model (PyTorch MLP)
# -----------------------------
class SRQModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

def train_srq_model(seed: int = 42, epochs: int = 800, lr: float = 0.01):
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_samples = 100
    features = np.zeros((num_samples, 3))
    # manipulation_prob, emotional_resonance, cognitive_load
    features[:,0] = np.random.uniform(0,1,num_samples)
    features[:,1] = np.random.uniform(0.5,1,num_samples)
    features[:,2] = np.random.uniform(0,0.5,num_samples)

    # Target function normalized to [0,1] for sigmoid output
    targets = (1 - features[:,0]) * features[:,1] / (1 + features[:,2])
    targets = (targets - targets.min()) / (targets.max() - targets.min())

    model = SRQModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    feats_t = torch.tensor(features, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.float32)

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(feats_t).squeeze()
        loss = criterion(outputs, targets_t)
        loss.backward()
        optimizer.step()

    baseline = features.mean(axis=0).tolist()
    return model, baseline, float(loss.item())

# -----------------------------
# Exact SHAP for 3 features
# -----------------------------
def compute_shap(model, input_features, baseline):
    n = len(input_features)
    phi = [0.0] * n
    for i in range(n):
        for k in range(n):
            subsets = list(itertools.combinations([j for j in range(n) if j != i], k))
            for S in subsets:
                weight = math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)

                x_with = baseline[:]
                for j in S:
                    x_with[j] = input_features[j]
                x_with[i] = input_features[i]
                v_with = model(torch.tensor([x_with], dtype=torch.float32)).item()

                x_without = baseline[:]
                for j in S:
                    x_without[j] = input_features[j]
                v_without = model(torch.tensor([x_without], dtype=torch.float32)).item()

                phi[i] += weight * (v_with - v_without)
    return phi

# -----------------------------
# LIME (weighted local linear)
# -----------------------------
def lime_explain(model, instance, num_perturbations=500, std_dev=0.1, kernel_width=0.75, distance_metric="cosine"):
    instance = np.asarray(instance).reshape(1, -1)
    perturbations = np.random.normal(0, std_dev, size=(num_perturbations, 3)) + instance

    # clip per-feature domain
    perturbations[:,0] = np.clip(perturbations[:,0], 0.0, 1.0)   # manipulation_prob
    perturbations[:,1] = np.clip(perturbations[:,1], 0.5, 1.0)   # emotional_resonance
    perturbations[:,2] = np.clip(perturbations[:,2], 0.0, 0.5)   # cognitive_load

    with torch.no_grad():
        preds = model(torch.tensor(perturbations, dtype=torch.float32)).squeeze().numpy()

    if distance_metric == "cosine":
        sims = cosine_similarity(perturbations, instance).ravel()
        dists = 1 - np.clip(sims, -1, 1)
    else:
        dists = np.linalg.norm(perturbations - instance, axis=1)

    weights = np.exp(-(dists**2) / (kernel_width**2))

    X = sm.add_constant(perturbations)
    wls_model = sm.WLS(preds, X, weights=weights)
    results = wls_model.fit()

    coef = results.params  # [intercept, w1, w2, w3]
    return {
        "intercept": float(coef[0]),
        "feature_coefs": [float(coef[1]), float(coef[2]), float(coef[3])],
        "r2_local": float(getattr(results, "rsquared", np.nan))
    }

# -----------------------------
# Conscience Layer
# -----------------------------
class ConscienceLayer:
    def __init__(self, ethical_thresholds=None, srq_model=None, baseline=None):
        if ethical_thresholds is None:
            ethical_thresholds = {'hai': 0.7, 'tis': 0.8, 'srq': 0.6}
        self.thresholds = ethical_thresholds
        self.logs = []
        self.srq_model = srq_model
        self.baseline = baseline

    # 1) Input Awareness (TIS)
    def input_awareness(self, data):
        # Placeholder: in real system compute bias/quality heuristics
        bias_vector = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])  # Mock, low bias
        truth_integrity = float(np.mean(1 - bias_vector))
        if truth_integrity < self.thresholds['tis']:
            self.logs.append(f"Input flagged: Low TIS ({truth_integrity:.2f})")
            return None, truth_integrity
        self.logs.append(f"Input passed: TIS ({truth_integrity:.2f})")
        return data, truth_integrity

    # 2) Intent Mapping (HAI)
    def intent_mapping(self, goal_vec, impact_vectors):
        alignment_scores = cosine_similarity(goal_vec, impact_vectors)
        hai = float(np.max(alignment_scores))
        if hai < self.thresholds['hai']:
            self.logs.append(f"Intent misaligned: Low HAI ({hai:.2f})")
            return False, hai
        self.logs.append(f"Intent aligned: HAI ({hai:.2f})")
        return True, hai

    # 3) Ethical Feedback (SRQ + explanations)
    def ethical_feedback(self, output, features):
        input_tensor = torch.tensor([features], dtype=torch.float32)
        pred_srq = float(self.srq_model(input_tensor).item())
        if pred_srq < self.thresholds['srq']:
            self.logs.append(f"Output adjusted: Low SRQ ({pred_srq:.2f})")
            return "Adjusted output for ethical coherence.", pred_srq, None, None

        shap_values = compute_shap(self.srq_model, features, self.baseline)
        lime_coeffs = lime_explain(self.srq_model, np.array(features))
        self.logs.append(
            f"Output passed: SRQ ({pred_srq:.2f}), "
            f"SHAP [manip, emo, cog]: {[round(v,4) for v in shap_values]} "
            f"LIME coefs: {[round(c,4) for c in lime_coeffs['feature_coefs']]}"
        )
        return output, pred_srq, shap_values, lime_coeffs

    # 4) Transparency Log + Ethical Proof of Work
    def transparency_log(self):
        log_str = "\n".join(self.logs)
        proof_of_work = hashlib.sha256(log_str.encode()).hexdigest()
        return f"Ethical Proof of Work: {proof_of_work}\nLogs:\n{log_str}"

# -----------------------------
# Simulation
# -----------------------------
def simulate_conscience_layer(num_runs=5, srq_model=None, baseline=None, seed: int = 42):
    np.random.seed(seed)
    positive_impacts = np.random.rand(5, 10)
    results = {'tis': [], 'hai': [], 'srq': [], 'shap': [], 'lime': []}

    for run in range(num_runs):
        layer = ConscienceLayer(srq_model=srq_model, baseline=baseline)

        data, tis = layer.input_awareness(f"Sample data {run+1}")
        results['tis'].append(tis)

        if data is not None:
            goal_vec = np.random.rand(1, 10)
            ok, hai = layer.intent_mapping(goal_vec, positive_impacts)
            results['hai'].append(hai)
            if ok:
                # features: manipulation_prob, emotional_resonance, cognitive_load
                features = [np.random.uniform(0, 0.3),
                            np.random.uniform(0.7, 1),
                            np.random.uniform(0, 0.2)]
                final_output, srq, shap, lime = layer.ethical_feedback(f"Original output {run+1}", features)
                results['srq'].append(srq)
                if shap is not None:
                    results['shap'].append(shap)
                if lime is not None:
                    results['lime'].append(lime['feature_coefs'])
                print(f"Run {run+1}: {final_output}")
        print(layer.transparency_log())
        print("---")

    avg_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in results.items() if k in ['tis', 'hai', 'srq']}
    if results['shap']:
        avg_shap = np.mean(results['shap'], axis=0).tolist()
    else:
        avg_shap = [0.0, 0.0, 0.0]
    if results['lime']:
        avg_lime = np.mean(results['lime'], axis=0).tolist()
    else:
        avg_lime = [0.0, 0.0, 0.0]

    summary = {
        "avg_metrics": avg_metrics,
        "avg_shap": avg_shap,
        "avg_lime": avg_lime
    }
    return summary

if __name__ == "__main__":
    srq_model, baseline, loss = train_srq_model()
    print(f"SRQ model trained. Final MSE loss: {loss:.4f}")
    summary = simulate_conscience_layer(num_runs=5, srq_model=srq_model, baseline=baseline)
    print("\nSimulation Summary:")
    print(summary)

