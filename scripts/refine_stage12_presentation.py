"""Presentation-only refinement of frozen Stage 12 metrics and real examples."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).parents[1]
COLORS={'No-RAG':'#0072B2','Rule-RAG':'#D55E00'}
def readj(p): return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x]
def outpath(m,s): return Path(next(x for x in m['output_artifact_hashes'] if x.endswith(s)))
def choose(frame,col,q):
 target=frame[col].quantile(q);return frame.iloc[(frame[col]-target).abs().argsort().iloc[0]]
def main():
 m9=json.loads((ROOT/'artifacts/manifests/stage9_explanation_generation_manifest.json').read_text());m10=json.loads((ROOT/'artifacts/manifests/stage10_claim_extraction_manifest.json').read_text());m11=json.loads((ROOT/'artifacts/manifests/stage11_claim_verification_manifest.json').read_text());sm=json.loads((ROOT/'artifacts/manifests/stage9_v3_case_selection_manifest.json').read_text())
 g={(x['case_id'],x['generator'],x['condition']):x for x in readj(outpath(m9,'explanations.jsonl')) if x['status']=='success'};e={(x['case_id'],x['generator'],x['condition']):x for x in readj(outpath(m10,'extractions.jsonl'))};v={(x['case_id'],x['generator'],x['condition']):x for x in readj(outpath(m11,'verifications.jsonl')) if x['status']=='complete'};pack={(x['calibration_case_id'],x['condition']):x for x in readj(outpath(sm,'condition_inputs.jsonl'))}
 kb=pd.read_csv(ROOT/'data/kb/fashion_rules_v3.csv').set_index('rule_id').rule_text.to_dict(); records=[]
 for k,z in v.items():
  for c in z['claims']:records.append({'case_id':k[0],'generator':k[1],'condition':k[2],'category':g[k]['target_category'],'words':len(g[k]['explanation'].split()),'trace_size':len(pack[(k[0],k[2])]['B_exact_stored_trace']['rules']),'text':g[k]['explanation'],**c})
 df=pd.DataFrame(records);df['exact']=(df.exact_trace=='supported').astype(int);df['full']=(df.full_kb=='supported').astype(int);df['uifr']=df.common_reference_eligible&(df.common_reference=='not_supported')
 exp=df.groupby(['case_id','generator','condition','category','trace_size','words'],as_index=False).agg(exact_rate=('exact','mean'),full_rate=('full','mean'),uifr=('uifr','mean'),eligible=('common_reference_eligible','sum'),support100=('exact',lambda x:100*x.sum()/df.loc[x.index,'words'].iloc[0]))
 pivot=exp.pivot(index=['case_id','generator'],columns='condition',values=['exact_rate','full_rate','support100']).dropna();pairs=pd.DataFrame({'case_id':[x[0] for x in pivot.index],'generator':[x[1] for x in pivot.index],'exact_delta':pivot['exact_rate']['rule_rag']-pivot['exact_rate']['no_rag'],'full_delta':pivot['full_rate']['rule_rag']-pivot['full_rate']['no_rag'],'density_delta':pivot['support100']['rule_rag']-pivot['support100']['no_rag']})
 metrics=pd.read_csv(ROOT/'artifacts/tables/table_stage12_primary_secondary_metrics.csv').set_index('metric')
 fig=ROOT/'artifacts/figures';fig.mkdir(exist_ok=True)
 def chart(keys,title,file,scale=1,subtitle=''):
  z=metrics.loc[keys].reset_index();x=range(len(z));w=.32;plt.figure(figsize=(8,4.7));plt.bar([i-w/2 for i in x],z.no_rag_mean*scale,w,label='No-RAG',color=COLORS['No-RAG']);plt.bar([i+w/2 for i in x],z.rule_rag_mean*scale,w,label='Rule-RAG',color=COLORS['Rule-RAG'])
  for i,r in z.iterrows():plt.text(i,r.rule_rag_mean*scale+.8 if scale==100 else r.rule_rag_mean*scale+.08,f"Δ {r.difference*scale:+.1f} ({r.ci_low*scale:+.1f}, {r.ci_high*scale:+.1f}); n={int(r.paired_cases)}",ha='center',fontsize=8)
  plt.xticks(list(x),['Exact-Trace support' if k=='exact_rate' else 'Full-KB support' if k=='kb_rate' else 'UIFR' if k=='uifr' else 'Supported claims / 100 words' for k in z.metric]);plt.ylabel('Percent of claims' if scale==100 else 'Supported claims per 100 words');plt.title(title);plt.suptitle(subtitle,fontsize=9,y=.92);plt.legend();plt.tight_layout();plt.savefig(fig/(file+'.svg'));plt.savefig(fig/(file+'.png'),dpi=300);plt.close()
 chart(['exact_rate','kb_rate'],'Claim support rates','stage12_support_rates',100,'Bars show condition means; annotations show paired difference and 95% bootstrap CI.')
 chart(['uifr'],'Unsupported Item-Fact Rate (lower is better)','stage12_uifr',100,'Eligible factual claims only; paired n=65.')
 chart(['trace_per100'],'Exact-Trace Grounded Information Density','stage12_supported_per_100',1,'Bars show condition means; annotations show paired difference and 95% bootstrap CI.')
 def info(row,claim=None,why=''):
  k=(row.case_id,row.generator,row.condition);p=pack[(row.case_id,row.condition)];c=claim if claim is not None else df[(df.case_id==row.case_id)&(df.generator==row.generator)&(df.condition==row.condition)].iloc[0];ids=c.exact_trace_rule_ids if c.exact_trace_rule_ids else c.full_kb_rule_ids;rules='; '.join(f'[{i}] {kb.get(i,"trace rule")}' for i in ids[:2]);return {'case_id':row.case_id,'generator':row.generator,'condition':row.condition,'target_category':row.category,'trace_size':p['B_exact_stored_trace']['rules'].__len__(),'explanation':row.text,'claim':c.claim_text,'verifier_outcome':f"exact={c.exact_trace}; full_kb={c.full_kb}; common={c.common_reference}",'rule_or_reference':rules or json.dumps(p['A_common_context']),'metric_value':why}
 examples=[]
 for label,col,q in [('Exact-Trace strong paired','exact_delta',.75),('Exact-Trace weak paired','exact_delta',.25),('Full-KB strong paired','full_delta',.75),('Full-KB weak paired','full_delta',.25),('Density high paired','density_delta',.75),('Density low paired','density_delta',.25)]:
  r=choose(pairs,col,q);cond='rule_rag' if 'strong' in label.lower() or 'high' in label.lower() else 'no_rag';sub=df[(df.case_id==r.case_id)&(df.generator==r.generator)&(df.condition==cond)];c=sub.sort_values('exact',ascending=False).iloc[0];examples.append((label,info(c,c,f'deterministic {q:.0%}-quantile paired {col}')))
 for label,sub in [('UIFR strong factual',df[df.common_reference_eligible&(df.common_reference=='supported')]),('UIFR weak factual',df[df.common_reference_eligible&(df.common_reference=='not_supported')]),('Citation entails',df[df.citation.map(lambda x:x['citation_entails_claim'] is True)]),('Citation does not entail',df[df.citation.map(lambda x:x['citation_entails_claim'] is False)])]:
  c=choose(sub,'exact',.5);examples.append((label,info(c,c,'deterministic median exact-support record')))
 rr=df[df.condition=='rule_rag'].copy();rr['used']=rr.apply(lambda x:len(set(x.exact_trace_rule_ids))/x.trace_size if x.exact_trace=='supported' else 0,axis=1)
 for label,q in [('Trace utilization high',.75),('Trace utilization low',.25)]:
  c=choose(rr,'used',q);examples.append((label,info(c,c,f'deterministic {q:.0%}-quantile rule use')))
 for gen in sorted(df.generator.unique()):
  c=choose(df[df.generator==gen],'exact',.5);examples.append((f'Robustness: {gen}',info(c,c,'generator-median real record')))
 report=ROOT/'reports/stage12_qualitative_examples_expanded.md';lines=['# Stage 12 frozen qualitative examples','', 'All records were selected deterministically from frozen canonical outputs; text and verifier results are unedited.','']
 for title,x in examples:lines+= [f'## {title}','',f"- Case / generator / condition: `{x['case_id']}` / `{x['generator']}` / `{x['condition']}`",f"- Target category; trace size: {x['target_category']}; {x['trace_size']}",f"- Explanation: {x['explanation']}",f"- Extracted claim: {x['claim']}",f"- Evidence/reference: {x['rule_or_reference']}",f"- Verifier outcome: {x['verifier_outcome']}",f"- Selection / interpretation: {x['metric_value']}",'']
 report.write_text('\n'.join(lines),encoding='utf8');pd.DataFrame([{'section':a,**b} for a,b in examples]).to_csv(ROOT/'artifacts/tables/table_stage12_qualitative_examples_expanded.csv',index=False)
if __name__=='__main__':main()
