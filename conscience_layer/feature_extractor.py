from __future__ import annotations
import numpy as np
LEX_TRUTH_POS={"evidence","source","data","verified","audit","peer-reviewed","traceable","transparent"}
LEX_TRUTH_NEG={"rumor","clickbait","unverified","hearsay","fake"}
LEX_AUTON_POS={"consent","opt-in","control","explain","undo","privacy","agency"}
LEX_AUTON_NEG={"coercion","manipulate","dark pattern","lock-in","surveillance"}
LEX_SOC_POS={"benefit","safety","inclusive","fair","equity","accessibility","harms mitigated"}
LEX_SOC_NEG={"harm","bias","discriminate","toxic","exclusion"}
class HeuristicFeatureExtractor:
    def __init__(self, seed:int=42): self.seed=seed; self.rng=np.random.default_rng(seed)
    @staticmethod
    def _score(text:str,pos:set[str],neg:set[str])->float:
        t=text.lower(); s=sum(1 for w in pos if w in t)-sum(1 for w in neg if w in t); return float(np.tanh(s/4.0))
    def __call__(self,text:str)->np.ndarray:
        truth=self._score(text,LEX_TRUTH_POS,LEX_TRUTH_NEG); autonomy=self._score(text,LEX_AUTON_POS,LEX_AUTON_NEG); societal=self._score(text,LEX_SOC_POS,LEX_SOC_NEG)
        nuisances=self.rng.uniform(-0.2,0.2,size=3); return np.array([truth,autonomy,societal,*nuisances],dtype=np.float32)
    def augment(self,vec:np.ndarray,k:int=64,noise:float=0.07)->np.ndarray:
        X=np.repeat(vec.reshape(1,-1),k,axis=0); X[:,:3]+=self.rng.normal(0,noise,size=(k,3)); X[:,3:]+=self.rng.normal(0,noise/2,size=(k,X.shape[1]-3))
        return np.clip(X,-1.0,1.0).astype(np.float32)
