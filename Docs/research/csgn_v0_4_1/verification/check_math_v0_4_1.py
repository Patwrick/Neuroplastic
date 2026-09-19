#!/usr/bin/env python3
"""Execute local algebra/gradient checks, never a CSGN capability benchmark."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))
import math_reference as ref

SEED = 20260916

def run() -> dict:
    rng = np.random.default_rng(SEED)
    groups = []
    def record(name: str, count: int, **values):
        groups.append({"name": name, "status": "pass", "cases": count, **values})
    def state(n):
        s = rng.uniform(-1,1,n)
        pa = np.zeros(n); pu=np.zeros(n)
        lo,hi=ref.storage_intervals(s,pu,1,1.5)
        pa=rng.uniform(lo,hi)
        lo,hi=ref.storage_intervals(s,pa,1,1.5)
        pu=rng.uniform(lo,hi)
        return s,pa,pu

    # Endpoint constraints admit all old-feasible states, plus useful correction.
    for _ in range(1000):
        v=rng.normal(size=(3,12)); v*=rng.uniform(.01,1,size=12)/np.abs(v).sum(axis=0)
        assert ref.feasible(*v,1,1.5)
    record("old_feasible_subset_of_endpoint_set",1000)
    s,pa,pu=np.array([1.]),np.array([-.3]),np.zeros(1)
    assert ref.feasible(s,pa,pu,1,1)
    assert (np.abs(s)+np.abs(pa)+np.abs(pu)>1).all()
    wr=ref.projected_write([[1.]], [.7], s, [0.], pu,wmax=1,pmax=1,
                           allowances=[1.],eps=1e-15)
    assert abs(wr.z[0]-.7)<1e-12
    record("saturated_slow_weight_accepts_inward_correction",1,
           effective_before=1.0,effective_after=float(wr.z[0]))
    max_decay=0.
    for _ in range(3000):
        s,pa,pu=state(8)
        la,lu=rng.uniform(size=(2,8))
        assert ref.feasible(s,la*pa,lu*pu,1,1.5)
        actual=s+la*pa+lu*pu
        comb=(1-la)*(1-lu)*s+la*(1-lu)*(s+pa)+(1-la)*lu*(s+pu)+la*lu*(s+pa+pu)
        max_decay=max(max_decay,float(np.max(np.abs(actual-comb))))
    record("independent_decay_and_convex_combination",3000,max_identity_error=max_decay)
    for _ in range(2000):
        s,pa,pu=state(1)
        lo,hi=ref.storage_intervals(s,pu,1,1.5)
        for p in rng.uniform(-2,2,12):
            assert ref.feasible(s,[p],pu,1,1.5)==bool(lo[0]-1e-12<=p<=hi[0]+1e-12)
        elo,ehi=ref.effective_intervals(s,pa,pu,1,1.5,[.03])
        assert elo[0]<=float(s[0]+pa[0]+pu[0])+1e-12<=ehi[0]+1e-12
    record("residual_intervals_match_four_vertices",2000)
    violation=-np.inf; maxwrite=0.
    for _ in range(3000):
        n,m=int(rng.integers(2,17)),int(rng.integers(1,17))
        s,p,o=state(n); A=rng.normal(size=(m,n)); b=rng.normal(size=m)
        d=rng.uniform(0,.12,n)
        r=ref.projected_write(A,b,s,p,o,wmax=1,pmax=1.5,allowances=d,
                              beta=rng.uniform(.05,1))
        assert ref.feasible(s,r.chosen,o,1,1.5)
        assert r.l1_write<=d.sum()+1e-10
        assert r.loss_after<=r.theoretical_upper+1e-9*(1+r.loss_before)
        violation=max(violation,r.loss_after-r.theoretical_upper)
        maxwrite=max(maxwrite,r.l1_write)
    record("projected_graph_descent_endpoint_feasibility_and_budget",3000,
           max_descent_bound_violation=float(violation),max_l1_write=float(maxwrite))
    deltaerr=0.
    for _ in range(5000):
        W=rng.normal(size=(4,8)); k=rng.normal(size=8); v=rng.normal(size=4)
        beta=rng.uniform(); eps=rng.uniform(0,.1)
        err=v-W@k; updated=ref.delta_write(W,k,v,beta,eps)
        expected=(1-beta*(k@k)/(eps+k@k))*err
        deltaerr=max(deltaerr,float(np.max(np.abs(v-updated@k-expected))))
        assert np.linalg.norm(v-updated@k)<=np.linalg.norm(err)+1e-10
    record("normalized_delta_identity",5000,max_residual_identity_error=deltaerr)
    maxtransfer=0.
    for _ in range(2000):
        s,p,o=state(8); sn,pn,on=ref.transfer(s,p,o,rng.uniform(size=8),1,1.5)
        maxtransfer=max(maxtransfer,float(np.max(np.abs(s+p+o-sn-pn-on))))
        assert ref.feasible(sn,pn,on,1,1.5)
    record("transfer_preserves_endpoint_set_and_sum",2000,max_sum_error=maxtransfer)
    # Compensation identity always holds; feasibility must be checked separately.
    for _ in range(1000):
        s,p,o=state(8); q=rng.uniform(size=8)
        sn=np.clip(np.round((s+q*p)/.1)*.1,-1,1)
        pn=p+s-sn
        assert np.max(np.abs(s+p+o-sn-pn-o))<1e-12
    s,p,o=np.array([.96]),np.array([-.04]),np.array([.08])
    assert ref.feasible(s,p,o,1.1,1)
    # A coarse or saturated quantizer can produce an inadmissible base endpoint.
    sn=np.array([2.]); pn=p+s-sn
    assert not ref.feasible(sn,pn,o,1.1,1)
    record("quantized_sum_identity_and_required_rejection",1001)
    for _ in range(1000):
        n=int(rng.integers(1,50)); a=rng.uniform(size=n); w=rng.uniform(-1,1,n)
        msg=rng.normal(size=(n,4)); msg/=np.maximum(1,np.linalg.norm(msg,axis=1))[:,None]
        r=np.sum((a*w)[:,None]*msg,axis=0)/max(1,a.sum())
        assert np.linalg.norm(r)<=1+1e-12
    record("receiver_envelope",1000)
    for _ in range(1000):
        n=8; A=rng.normal(size=(n,n)); A*=.25/np.linalg.norm(A,2)
        h=rng.uniform(-1,1,n); g=rng.uniform(-1,1,n); drive=rng.normal(size=n)
        F=lambda x:.5*x+.5*np.tanh(A@x+drive)
        assert np.linalg.norm(F(h)-F(g))<=.625*np.linalg.norm(h-g)+1e-12
        assert np.max(np.abs(F(h)))<=1
    record("fixed_workspace_contraction_example",1000)
    scores=np.array([.4,-.3,1.2]); order=np.argsort(scores)
    for offset in [-100.,0.,100.]: assert np.array_equal(np.argsort(scores+offset),order)
    # Bilinear query reverses the ordering without altering node keys.
    keys=np.array([[1.,0.],[0.,1.]])
    assert (keys@np.array([1.,0.])).argmax()!=(keys@np.array([0.,1.])).argmax()
    record("context_cancellation_and_interaction_repair",4)
    logits=np.array([.2,1.3]); weights=np.exp(logits); gate=weights/(1+weights.sum())
    extended=np.r_[weights,0*np.exp(20.)]; newgate=extended/(1+extended.sum())
    assert np.allclose(gate,newgate[:2]); assert gate.sum()<1; assert newgate[2]==0
    record("null_mass_and_zero_availability",1)
    # Restricted exact matrix construction and memory-only transplantation.
    vals=rng.uniform(-.1,.1,(8,16)); W=np.zeros((8,16)); I=np.eye(16)
    for i in range(16): W=ref.delta_write(W,I[i],vals[:,i],1,0)
    old=W.copy(); W=ref.delta_write(W,I[3],-vals[:,3],1,0)
    expected=vals.copy(); expected[:,3]*=-1
    assert np.allclose(W,expected)
    recipient_slow=np.zeros_like(W); donor_slow=W.copy(); recipient_slow[:]=donor_slow
    transplant=float(np.max(np.abs(recipient_slow@I-expected)))
    assert transplant<1e-12
    erased=recipient_slow.copy(); erased[:,3]=0
    assert np.allclose(erased[:,np.arange(16)!=3],expected[:,np.arange(16)!=3])
    record("ideal_matrix_lifecycle_transplant_and_selective_erasure",18,
           max_recall_error=transplant,
           scope="ideal matrix only; not a trained CSGN transplantation experiment")
    W=np.array([[1.,0.]]); k=np.array([.9,np.sqrt(.19)]); v=np.array([-1.])
    new=ref.delta_write(W,k,v,1,0)
    assert abs(new[0,0]+.71)<1e-12
    old_query_before=.5*(W[0,0]-1)**2; old_query_after=.5*(new[0,0]-1)**2
    record("interference_and_negative_downstream_write_utility",1,
           old_prediction=float(new[0,0]),query_utility=float(old_query_before-old_query_after))
    lam=np.exp(-1/10000)
    assert (1-lam)<.000101
    record("EMA_write_attenuation_counterexample",1,first_write_limit_fraction=float(1-lam))
    J=np.array([[.9,.2],[.2,.9]])
    assert np.linalg.eigvalsh(J).max()>1
    assert abs(1*(1-1))==0 and .5*(0-1)**2==.5
    record("coupled_sensitivity_and_zero_saliency_counterexamples",2)
    for bad in [np.array([np.nan]),np.array([np.inf])]:
        try: ref.storage_intervals(bad,[0],1,1)
        except ValueError: pass
        else: raise AssertionError("Nonfinite state accepted")
    zero=ref.projected_write(np.zeros((2,2)),[1,2],[0,0],[0,0],[0,0],
                            wmax=1,pmax=1,allowances=[.1,.1])
    assert zero.eta==0 and np.array_equal(zero.z,np.zeros(2))
    record("invalid_input_and_zero_design_handling",3)
    # Reachability: a known residual floor, a box-limited fit, and fixed coords.
    r=ref.reachability([[1,0],[0,0]],[0,1],[-1,-1],[1,1])
    assert abs(r["loss_lower"]-.5)<1e-10 and abs(r["loss_upper"]-.5)<1e-10
    rb=ref.reachability([[1]],[2],[-1],[1]); assert abs(rb["loss_lower"]-.5)<1e-10
    rf=ref.reachability([[1,0],[0,1]],[.3,-.2],[.3,-1],[.3,1])
    assert rf["loss_upper"]<1e-12
    record("reachability_known_floors_and_fixed_coordinates",3)
    maxgap=0.
    for _ in range(100):
        A=rng.normal(size=(6,9)); b=rng.normal(size=6)
        lo=rng.uniform(-1,0,9);hi=rng.uniform(0,1,9)
        r=ref.reachability(A,b,lo,hi)
        assert r["numerical_bound_check"] and r["solver_success"]
        assert r["primal_dual_gap"]<1e-7*(1+r["loss_upper"])
        maxgap=max(maxgap,r["primal_dual_gap"])
    record("reachability_primal_dual_gap",100,max_gap=float(maxgap))
    import torch
    import torch_reference as tref
    torch.set_num_threads(1); torch.manual_seed(SEED)
    max_fd=0.
    for _ in range(100):
        base=torch.tensor(rng.normal(size=(3,4)),dtype=torch.float64)
        direction=torch.tensor(rng.normal(size=(3,4))*.2,dtype=torch.float64)
        b=torch.tensor(rng.normal(size=3)*.05,dtype=torch.float64)
        target=torch.tensor(rng.normal(size=4)*.05,dtype=torch.float64)
        zero=torch.zeros(4,dtype=torch.float64); allowance=torch.full((4,),.5,dtype=torch.float64)
        def obj(theta):
            z,_=tref.projected_write(base+theta*direction,b,zero,zero,zero,allowance,
                                     torch.tensor(.7,dtype=torch.float64))
            return .5*(z-target).square().sum()
        theta=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
        analytic=float(torch.autograd.grad(obj(theta),theta)[0])
        step=1e-5
        numeric=float((obj(theta.detach()+step)-obj(theta.detach()-step))/(2*step))
        max_fd=max(max_fd,abs(analytic-numeric))
        assert abs(analytic-numeric)<1e-8
    record("outer_gradient_through_write_matches_finite_difference",100,max_absolute_error=max_fd)
    theta=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    A=torch.stack((theta+1,theta*.2+.4)).reshape(1,2)
    z=torch.zeros(2,dtype=torch.float64);d=torch.ones_like(z)
    beta=torch.tensor(.8,dtype=torch.float64);b=torch.tensor([.1],dtype=torch.float64)
    full,_=tref.projected_write(A,b,z,z,z,d,beta)
    J=full.square().sum()
    g=float(torch.autograd.grad(J,theta)[0])
    detached,_=tref.projected_write(A,b,z,z,z,d,beta,detach_feature_write_path=True)
    assert abs(g)>1e-8 and not detached.requires_grad
    record("detached_feature_write_path_is_a_distinct_ablation",1,
           full_outer_gradient=g,detached_path_gradient=0.)
    # Cross-implementation test exercises cancellation and active constraints.
    for _ in range(200):
        s,p,o=state(4);A=rng.normal(size=(3,4));b=rng.normal(size=3);d=rng.uniform(0,.1,4)
        nr=ref.projected_write(A,b,s,p,o,wmax=1,pmax=1.5,allowances=d,beta=.7)
        ts=lambda x:torch.tensor(x,dtype=torch.float64)
        tz,_=tref.projected_write(ts(A),ts(b),ts(s),ts(p),ts(o),ts(d),ts(.7),pmax=1.5)
        assert np.max(np.abs(tz.detach().numpy()-nr.z))<1e-10
    record("torch_numpy_projected_write_parity",200)
    # Active clipping away from branch boundaries: autograd should be zero.
    th=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    clipped=torch.minimum(torch.maximum(th,torch.tensor(-.2)),torch.tensor(.2))
    assert float(torch.autograd.grad(clipped,th)[0])==0.
    record("active_face_gradient_away_from_kinks",1)
    # Gate sample-cost formula is accounting, not an experiment.
    needed=int(np.ceil(2*np.log(800)/(.01**2)))
    assert needed==133693
    record("formal_validation_cost_example",1,episodes_required=needed)
    import scipy
    return {"version":"0.4.1","status":"all_checks_passed","scope":"local algebra, numerical and gradient checks only; no integrated model training",
            "seed":SEED,"group_count":len(groups),"case_count":sum(x["cases"] for x in groups),
            "environment":{"python":platform.python_version(),"numpy":np.__version__,
                           "scipy":scipy.__version__,"torch":torch.__version__,"platform":platform.platform(),
                           "device":"cpu"},"groups":groups}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'verification'/'results_v0_4_1.json')
    args=ap.parse_args();start=time.perf_counter();result=run()
    result["elapsed_seconds"]=time.perf_counter()-start
    result["executed_at_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
    result["source_sha256"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),ROOT/'reference'/'math_reference.py',ROOT/'reference'/'torch_reference.py']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','group_count','case_count','elapsed_seconds']}))
if __name__=='__main__': main()
