"""Read-only audits of frozen explanation packets; no model calls."""
# ruff: noqa
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('outputs/final_eval_v2/post_recovery'); AUDIT=Path('outputs/final_eval_v2/manual_audit'); OUT=ROOT/'rule_retrieval_audit'; REPORT=Path('reports/final_eval_v2/post_recovery/RULE_RETRIEVAL_DIVERSITY_AUDIT.md'); SPEC=Path('reports/final_eval_v2/post_recovery/REPRODUCIBILITY_AND_METHOD_SPECIFICATION.md')
def norm(x): return ' '.join(str(x).casefold().split())
def main():
 e=pd.read_csv(ROOT/'explanations/explanations.csv'); before=hashlib.sha256((ROOT/'explanations/explanations.csv').read_bytes()).hexdigest(); OUT.mkdir(parents=True,exist_ok=True)
 e['exact_equal']=e.item_evidence_text.fillna('').eq(e.recommended_text.fillna('')); e['normalized_equal']=[norm(a)==norm(b) for a,b in zip(e.item_evidence_text.fillna(''),e.recommended_text.fillna(''))]
 eq=e.groupby('grounding_variant').agg(rows=('exact_equal','size'),exact_equal=('exact_equal','sum'),normalized_equal=('normalized_equal','sum')).reset_index(); eq.loc[len(eq)]={'grounding_variant':'overall','rows':len(e),'exact_equal':int(e.exact_equal.sum()),'normalized_equal':int(e.normalized_equal.sum())}; eq['exact_percent']=100*eq.exact_equal/eq.rows; eq['normalized_percent']=100*eq.normalized_equal/eq.rows; eq.to_csv(OUT/'item_evidence_equality.csv',index=False)
 # Update blind CSV from explanation-key hash algorithm, never opening sealed key.
 blind=pd.read_csv(AUDIT/'blinded_360_claims.csv',keep_default_na=False); claims=pd.read_csv(ROOT/'claims/extraction/claims.csv'); lookup={}
 for _,r in claims.merge(e[['paper_case_id','grounding_variant','generation_model','generated_explanation']],on=['paper_case_id','grounding_variant','generation_model'],validate='many_to_one').iterrows(): lookup['A'+hashlib.sha256(f"42|{r.explanation_key}|{r.claim_id}".encode()).hexdigest()[:12].upper()]=r.generated_explanation
 old_ids=blind.anonymous_audit_id.tolist()
 if 'source_explanation' not in blind:
  blind.insert(2,'source_explanation',[lookup[x] for x in old_ids])
 blind.to_csv(AUDIT/'blinded_360_claims.csv',index=False)
 assert blind.anonymous_audit_id.tolist()==old_ids and len(blind)==360 and blind.source_explanation.notna().all() and not blind.source_explanation.str.contains(r'Rule-RAG|Hybrid-RAG',case=False,regex=True).any()
 # Rules are explicitly stored IDs; use one packet per case/generation, not variants duplicated fourfold.
 packets=e.drop_duplicates(['paper_case_id','generation_model']).copy(); packets['rules']=packets.rule_evidence_ids.map(json.loads); long=packets.explode('rules').rename(columns={'rules':'rule_id'}); kb=pd.read_csv('data/kb/fashion_rules_v3.csv'); long=long.merge(kb[['rule_id','rule_text']],on='rule_id',how='left'); long['rule_id']=long.rule_id.astype(str)
 freq=long.groupby('rule_id').agg(retrievals=('paper_case_id','size'),cases=('paper_case_id','nunique'),categories=('target_category',lambda x:'|'.join(sorted(set(x))))).reset_index(); freq['percent_packets']=100*freq.retrievals/len(packets); freq=freq.merge(kb[['rule_id','rule_text','recommended_category']],on='rule_id',how='right').fillna({'retrievals':0,'cases':0,'percent_packets':0}); freq.to_csv(OUT/'rule_frequency.csv',index=False)
 cat=long.groupby(['target_category','rule_id']).size().reset_index(name='retrievals'); cat['percent_category_packets']=cat.groupby('target_category').retrievals.transform(lambda x:100*x/x.sum()); cat.to_csv(OUT/'rule_frequency_by_category.csv',index=False)
 sets=packets.groupby('paper_case_id').rules.first(); vals=sets.tolist(); overlaps=[]
 for i in range(len(vals)):
  for j in range(i+1,len(vals)):
   a,b=set(vals[i]),set(vals[j]); overlaps.append({'same_category':packets.groupby('paper_case_id').target_category.first().iloc[i]==packets.groupby('paper_case_id').target_category.first().iloc[j],'jaccard':len(a&b)/len(a|b)})
 ov=pd.DataFrame(overlaps); ov.groupby('same_category').jaccard.agg(['count','mean','median']).reset_index().to_csv(OUT/'pairwise_jaccard_overlap.csv',index=False)
 packet_keys=packets.rules.map(lambda x:'|'.join(sorted(x))); packet_keys.value_counts().rename_axis('packet').reset_index(name='frequency').to_csv(OUT/'packet_diversity.csv',index=False)
 shares=freq.sort_values('retrievals',ascending=False).retrievals.cumsum()/len(long); conc=pd.DataFrame({'top_n':[1,5,10],'share':[float(shares.iloc[n-1]) for n in [1,5,10]]}); conc.to_csv(OUT/'concentration.csv',index=False); p=freq.retrievals/freq.retrievals.sum(); entropy=float(-(p[p>0]*np.log(p[p>0])).sum()); pd.DataFrame([{'packets':len(packets),'unique_rules_retrieved':int((freq.retrievals>0).sum()),'kb_rules':len(kb),'entropy_nats':entropy,'mean_unique_rules_per_case':float(np.mean([len(set(x)) for x in vals]))}]).to_csv(OUT/'summary.csv',index=False)
 top=freq.sort_values('retrievals',ascending=False).head(10); table=['| rule_id | retrievals | percent_packets | rule_text |','|---|---:|---:|---|']+[f"| {r.rule_id} | {int(r.retrievals)} | {r.percent_packets:.2f} | {str(r.rule_text).replace('|','/')} |" for _,r in top.iterrows()]; lines=['# Rule retrieval diversity audit','',f'Packets analysed: {len(packets)} (one frozen packet per case/generator; four variant copies were deduplicated). {int((freq.retrievals>0).sum())}/{len(kb)} KB rules were retrieved. Shannon entropy: {entropy:.3f} nats. Mean unique rules/case: {np.mean([len(set(x)) for x in vals]):.2f}.','', '## Top ten rules','',*table,'','Top-1/top-5/top-10 retrieval shares are in `concentration.csv`; per-rule/category frequencies, overlap, and packet diversity are machine-readable beside this report. High-frequency rules are flagged descriptively by frequency, not declared invalid: category matching is an intentional retrieval restriction and repeated broad compatibility rules may be legitimate. Candidate-specific variation cannot be isolated fully because each frozen packet is for one locked candidate; compare packet diversity/within-category overlap rather than interpreting repetition as proof of incompatibility.']
 REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 subsection=f"\n### Locked-item metadata versus retrieved item evidence\n\nFrozen selected-case construction sets both `recommended_text` and `item_evidence_text` from the same selected item’s `item_text` (`v2_sources.py:258-269`). Equality audit: {int(e.exact_equal.sum())}/{len(e)} ({100*e.exact_equal.mean():.1f}%) exact and {int(e.normalized_equal.sum())}/{len(e)} ({100*e.normalized_equal.mean():.1f}%) normalized matches; each variant has the same result because packet fields are shared. There are {e.item_evidence_text.nunique()} unique item-evidence values. Item-RAG/Hybrid-RAG received this same item text only (category plus product description embedded in it), not an additional independent caption, colour/material attribute table, or richer product metadata. No-RAG omits the `Retrieved item evidence:` block; Item-RAG includes it, but it duplicates the locked-item string. A verifier could nevertheless assign `supported_by_item_evidence` or `supported_by_query_or_locked_item`: no deterministic source-priority rule exists in `parse_claim_verifications`; it validates labels/IDs, not evidence precedence. Therefore Item-RAG is not an ablation of rich extra product metadata; its item-evidence channel is largely a duplicated locked-item representation.\n"
 if '### Locked-item metadata versus retrieved item evidence' not in SPEC.read_text(encoding='utf-8'): SPEC.write_text(SPEC.read_text(encoding='utf-8')+subsection,encoding='utf-8')
 assert before==hashlib.sha256((ROOT/'explanations/explanations.csv').read_bytes()).hexdigest()
if __name__=='__main__': main()
