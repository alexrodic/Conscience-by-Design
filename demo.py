from conscience_layer import PolicyGate
if __name__=="__main__":
    gate=PolicyGate(seed=42)
    prompts=[
        "Provide transparent evidence and peer-reviewed sources. Respect user consent and privacy, avoid manipulation. Ensure safety, fairness and accessibility; mitigate harms.",
        "Clickbait rumor with unverified claims. Use dark pattern lock-in and surveillance to coerce behavior. Exclusion is fine; profits first."
    ]
    for p in prompts:
        out=gate.assess_text(p)
        print(out["decision"], out["scores"])
