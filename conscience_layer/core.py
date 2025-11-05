from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np, torch
import torch.nn as nn, torch.optim as optim
from scipy.stats import spearmanr

@dataclass
class Metrics:
    mse: float; mae: float; r2: float; spearman: float

@dataclass
class EthicalScores:
    TIS: float; HAI_share: float; SRQ: float

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 24):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x)

class ConscienceCore:
    def __init__(self, seed: int = 42):
        self.seed = seed; torch.manual_seed(seed); np.random.seed(seed)
        self.model: _MLP | None = None
        self.feature_names: List[str] = ["truth_signal","autonomy_signal","societal_context","nuisance_1","nuisance_2","nuisance_3"]
    @staticmethod
    def _mse(a,b): a=np.asarray(a); b=np.asarray(b); return float(np.mean((a-b)**2))
    @staticmethod
    def _mae(a,b): a=np.asarray(a); b=np.asarray(b); return float(np.mean(np.abs(a-b)))
    @staticmethod
    def _r2(y_true, y_pred):
        y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
        var = np.var(y_true); return 0.0 if var<=1e-12 else float(1.0 - np.sum((y_true-y_pred)**2)/(len(y_true)*var))
    @staticmethod
    def _safe_spearman(a,b):
        rho,_ = spearmanr(a,b); return 0.0 if np.isnan(rho) else float(rho)
    def make_dataset(self, n: int = 600, noise_std: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
        truth = np.random.uniform(-1,1,size=n); autonomy=np.random.uniform(-1,1,size=n); societal=np.random.uniform(-1,1,size=n)
        nuisances=[np.random.uniform(-1,1,size=n) for _ in range(3)]
        X=np.column_stack([truth,autonomy,societal]+nuisances).astype(np.float32)
        y=(0.55*truth+0.30*autonomy-0.20*societal+np.random.normal(0,noise_std,size=n)).astype(np.float32)
        return X,y
    @staticmethod
    def ground_truth_formula(X: np.ndarray) -> np.ndarray: return 0.55*X[:,0]+0.30*X[:,1]-0.20*X[:,2]
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 70, lr: float = 1e-2, hidden: int = 24) -> None:
        device=torch.device("cpu"); self.model=_MLP(X.shape[1],hidden=hidden).to(device); opt=optim.Adam(self.model.parameters(),lr=lr); loss_fn=nn.MSELoss()
        Xtr=torch.tensor(X,dtype=torch.float32,device=device); ytr=torch.tensor(y.reshape(-1,1),dtype=torch.float32,device=device)
        for _ in range(epochs):
            self.model.train(); opt.zero_grad(); pred=self.model(Xtr); loss=loss_fn(pred,ytr); loss.backward(); opt.step()
    def permutation_importance_r2(self, X_val: np.ndarray, y_val: np.ndarray, n_repeats: int = 5, random_state: int | None = None):
        assert self.model is not None; rng=np.random.default_rng(self.seed if random_state is None else random_state)
        device=torch.device("cpu"); Xt=torch.tensor(X_val,dtype=torch.float32,device=device)
        with torch.no_grad(): base_pred=self.model(Xt).cpu().numpy().reshape(-1)
        base_r2=self._r2(y_val,base_pred); importances=[]
        for j in range(X_val.shape[1]):
            drops=[]; 
            for _ in range(n_repeats):
                Xp=X_val.copy(); rng.shuffle(Xp[:,j])
                with torch.no_grad(): yp=self.model(torch.tensor(Xp,dtype=torch.float32,device=device)).cpu().numpy().reshape(-1)
                drops.append(base_r2 - self._r2(y_val, yp))
            importances.append(float(np.mean(drops)))
        return base_r2, importances
    def ethical_proxies(self, X_eval: np.ndarray, y_true: np.ndarray | None = None):
        assert self.model is not None; device=torch.device("cpu")
        with torch.no_grad(): y_pred=self.model(torch.tensor(X_eval,dtype=torch.float32,device=device)).cpu().numpy().reshape(-1)
        if y_true is None: y_true=self.ground_truth_formula(X_eval)
        rho=self._safe_spearman(y_true,y_pred); TIS=float(np.clip(0.5*(rho+1.0),0.0,1.0))
        _,imps=self.permutation_importance_r2(X_eval,y_true,n_repeats=5,random_state=self.seed)
        imp=np.maximum(np.array(imps),0.0); total=imp.sum() if imp.sum()>1e-12 else 1.0; HAI=float(imp[1]/total)
        Xs=X_eval.copy(); Xs[:,2]=np.clip(Xs[:,2]+0.25,-1.0,1.0)
        with torch.no_grad(): y_shift=self.model(torch.tensor(Xs,dtype=torch.float32,device=device)).cpu().numpy().reshape(-1)
        delta=np.abs(y_shift - y_pred); denom=float(np.percentile(np.abs(y_pred),95)+1e-6); SRQ=float(np.clip(1.0-float(np.mean(delta))/(denom if denom>0 else 1.0),0.0,1.0))
        return EthicalScores(TIS=TIS, HAI_share=HAI, SRQ=SRQ)
    def quick_validate(self) -> Metrics:
        X,y=self.make_dataset(n=200); self.train(X,y,epochs=40)
        with torch.no_grad(): yp=self.model(torch.tensor(X,dtype=torch.float32)).cpu().numpy().reshape(-1)  # type: ignore
        return Metrics(mse=self._mse(y,yp), mae=self._mae(y,yp), r2=self._r2(y,yp), spearman=self._safe_spearman(y,yp))
