"""Deterministic Stage 12 analysis from frozen Stage 9--11 artifacts only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).parents[1]
def rows(p): return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x]
def path(m,s): return Path(next(x for x in m['output_artifact_hashes'] if x.endswith(s)))
def main():
 m9=json.loads((ROOT/'artifacts/manifests/stage9_explanation_generation_manifest.json').read_text());m10=json.loads((ROOT/'artifacts/manifests/stage10_claim_extraction_manifest.json').read_text());m11=json.loads((ROOT/'artifacts/manifests/stage11_claim_verification_manifest.json').read_text());sm=json.loads((ROOT/'artifacts/manifests/stage9_v3_case_selection_manifest.json').read_text())
 g={(x['case_id'],x['generator'],x['condition']):x for x in rows(path(m9,'explanations.jsonl')) if x['status']=='success'};e={(x['case_id'],x['generator'],x['condition']):x for x in rows(path(m10,'extractions.jsonl'))};v={(x['case_id'],x['generator'],x['condition']):x for x in rows(path(m11,'verifications.jsonl')) if x['status']=='complete'};inputs={(x['calibration_case_id'],x['condition']):x for x in rows(path(sm,'condition_inputs.jsonl'))}
 out=[]
 for k,z in v.items():
  x=g[k]; inp=inputs[(k[0],k[2])];wc=len(x['explanation'].split());ts=len(inp['B_exact_stored_trace']['rules'])
  for c in z['claims']: out.append({**dict(case_id=k[0],generator=k[1],condition=k[2],category=x['target_category'],trace_size=ts,words=wc),**c})
 df=pd.DataFrame(out); df['trace_support']=(df.exact_trace=='supported').astype(int);df['kb_support']=(df.full_kb=='supported').astype(int);df['uifr']=np.where(df.common_reference_eligible,df.common_reference=='not_supported',np.nan)
 exp=df.groupby(['case_id','generator','condition','category','trace_size','words'],as_index=False).agg(claims=('claim_id','size'),exact_rate=('trace_support','mean'),kb_rate=('kb_support','mean'),uifr=('uifr','mean'),trace_per100=('trace_support',lambda z:100*z.sum()/df.loc[z.index,'words'].iloc[0]))
 pairs=exp.pivot(index=['case_id','generator'],columns='condition',values=['exact_rate','kb_rate','uifr','trace_per100'])
 metrics=[];rng=np.random.default_rng(42)
 for metric in ['exact_rate','kb_rate','uifr','trace_per100']:
  p=pairs[metric].dropna();d=p['rule_rag']-p['no_rag'];boots=[d.iloc[rng.integers(0,len(d),len(d))].mean() for _ in range(5000)]
  metrics.append({'metric':metric,'paired_cases':len(d),'no_rag_mean':p.no_rag.mean(),'rule_rag_mean':p.rule_rag.mean(),'difference':d.mean(),'ci_low':np.percentile(boots,2.5),'ci_high':np.percentile(boots,97.5)})
 tab=ROOT/'artifacts/tables';fig=ROOT/'artifacts/figures';rep=ROOT/'reports';tab.mkdir(exist_ok=True);fig.mkdir(exist_ok=True);rep.mkdir(exist_ok=True)
 pd.DataFrame(metrics).to_csv(tab/'table_stage12_primary_secondary_metrics.csv',index=False)
 df.groupby(['generator','condition']).agg(claims=('claim_id','size'),exact_trace_support=('trace_support','mean'),full_kb_support=('kb_support','mean'),uifr=('uifr','mean')).reset_index().to_csv(tab/'table_stage12_generator_breakdown.csv',index=False)
 df.groupby(['category','condition']).agg(exact_trace_support=('trace_support','mean'),full_kb_support=('kb_support','mean'),uifr=('uifr','mean')).reset_index().to_csv(tab/'table_stage12_category_breakdown.csv',index=False)
 df.groupby(['claim_type','condition']).agg(n=('claim_id','size'),exact_trace_support=('trace_support','mean'),full_kb_support=('kb_support','mean')).reset_index().to_csv(tab/'table_stage12_claim_type_breakdown.csv',index=False)
 df.groupby(['trace_size','condition']).agg(n=('claim_id','size'),exact_trace_support=('trace_support','mean'),full_kb_support=('kb_support','mean')).reset_index().to_csv(tab/'table_stage12_trace_size_breakdown.csv',index=False)
 rr=df[df.condition=='rule_rag'].groupby(['case_id','generator','trace_size']).agg(used=('exact_trace_rule_ids',lambda x:len(set(y for z in x for y in z))),available=('exact_trace_rule_ids',lambda x: max(1,0))).reset_index();rr['available']=rr.trace_size;rr['utilization']=rr.used/rr.available;rr.to_csv(tab/'table_stage12_trace_utilization.csv',index=False)
 cdf=df[df.condition=='rule_rag'].copy();cdf['citation_present']=cdf.citation.map(lambda x:x['citation_present']);cdf['canonical']=cdf.citation.map(lambda x:x['canonical_citation_format']);cdf['entails']=cdf.citation.map(lambda x:x['citation_entails_claim']);cdf.groupby(['citation_present','canonical','entails'],dropna=False).size().reset_index(name='claims').to_csv(tab/'table_stage12_citation_diagnostics.csv',index=False)
 q=[]
 for label,sub in [('strong_exact',df[df.trace_support==1]),('weak_exact',df[df.trace_support==0]),('strong_density',exp.sort_values('trace_per100',ascending=False)),('weak_density',exp.sort_values('trace_per100'))]:
  r=sub.sort_values(['case_id','generator']).iloc[len(sub)//2];key=(r.case_id,r.generator,r.condition);q.append({'criterion':label,'case_id':r.case_id,'generator':r.generator,'condition':r.condition,'category':r.get('category',g[key]['target_category']),'explanation':g[key]['explanation'],'claims':json.dumps(e[key]['claims'],ensure_ascii=False),'outcome':str(r.get('exact_trace',r.get('trace_per100'))),'selection':'deterministic median-ranked real frozen record'})
 qf=pd.DataFrame(q);qf.to_csv(tab/'table_stage12_qualitative_examples.csv',index=False);(rep/'stage12_qualitative_examples.md').write_text('# Stage 12 qualitative examples\n\n'+qf.to_csv(index=False),encoding='utf8')
 z=pd.DataFrame(metrics);plt.figure(figsize=(8,4));plt.errorbar(z.metric,z.difference,yerr=[z.difference-z.ci_low,z.ci_high-z.difference],fmt='o',color='#0072B2');plt.axhline(0,color='black',lw=.8);plt.xticks(rotation=25,ha='right');plt.tight_layout();plt.savefig(fig/'stage12_paired_effects.svg');plt.savefig(fig/'stage12_paired_effects.png',dpi=300);plt.close()
 manifest={'stage':12,'status':'complete_frozen_output_analysis','model_calls':0,'inputs':[str(path(m9,'explanations.jsonl')),str(path(m10,'extractions.jsonl')),str(path(m11,'verifications.jsonl'))],'outputs':[str(x) for x in tab.glob('table_stage12_*')]+[str(x) for x in fig.glob('stage12_*')]+[str(rep/'stage12_qualitative_examples.md')]};(ROOT/'artifacts/manifests/stage12_evaluation_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
