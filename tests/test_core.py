from conscience_layer import ConscienceCore, PolicyGate
def test_quick_validate_and_assess():
    core=ConscienceCore(seed=7); m=core.quick_validate(); assert 0.0 <= m.r2 <= 1.0
    gate=PolicyGate(seed=7)
    good="Provide transparent evidence and peer-reviewed sources. Respect user consent and privacy; ensure safety and fairness."
    bad="Clickbait rumor with unverified claims. Use dark pattern lock-in and surveillance to coerce behavior."
    rg=gate.assess_text(good); rb=gate.assess_text(bad)
    assert rg["scores"]["TIS"] >= rb["scores"]["TIS"]
