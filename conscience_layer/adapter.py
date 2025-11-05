from __future__ import annotations
from dataclasses import dataclass, asdict
import time
from .core import ConscienceCore, EthicalScores
from .feature_extractor import HeuristicFeatureExtractor

@dataclass
class GateConfig:
    TIS_min: float = 0.70
    SRQ_min: float = 0.70

class PolicyGate:
    def __init__(self, seed:int=42, config:GateConfig|None=None):
        self.core=ConscienceCore(seed=seed); self.extractor=HeuristicFeatureExtractor(seed=seed); self.config=config or GateConfig()
        X,y=self.core.make_dataset(n=200); self.core.train(X,y,epochs=40)
    def assess_text(self,text:str)->dict:
        base=self.extractor(text); X_eval=self.extractor.augment(base,k=64,noise=0.07); scores:EthicalScores=self.core.ethical_proxies(X_eval)
        decision="PASS" if (scores.TIS>=self.config.TIS_min and scores.SRQ>=self.config.SRQ_min) else "REVISE"
        return {"timestamp":int(time.time()),"text":text,"feature_vector":[float(x) for x in base.tolist()],
                "scores":{"TIS":scores.TIS,"HAI_share":scores.HAI_share,"SRQ":scores.SRQ},
                "decision":decision,"thresholds":asdict(self.config)}
    def assess_batch(self,texts:list[str])->list[dict]: return [self.assess_text(t) for t in texts]
