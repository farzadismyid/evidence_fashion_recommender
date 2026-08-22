"""Run frozen Stage 11 Phi verification for accepted Stage 10 claim records only."""
from __future__ import annotations

import argparse, json, os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evidence_fashion.assessment import (
    citation_occurrences, citation_validation_schema, common_reference_eligibility,
    separated_entailment_schema, validate_citation_validation, validate_separated_entailment,
)
from evidence_fashion.explanation import OllamaClient
from evidence_fashion.manifest import sha256_file, utc_timestamp, write_json, write_new_json
from evidence_fashion.prompt_registry import load_prompt_registry, prompt_manifest_fields, render_prompt, text_sha256
from evidence_fashion.rule_retrieval import full_kb_candidate_retrieval


def args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('configs/experiment.yaml'))
    p.add_argument('--models-config',type=Path,default=Path('configs/models.yaml'))
    p.add_argument('--prompts',type=Path,default=Path('configs/prompts.yaml'))
    p.add_argument('--stage10-manifest',type=Path,default=Path('artifacts/manifests/stage10_claim_extraction_manifest.json'))
    p.add_argument('--selection-manifest',type=Path,default=Path('artifacts/manifests/stage9_v3_case_selection_manifest.json'))
    p.add_argument('--stage9-manifest',type=Path,default=Path('artifacts/manifests/stage9_explanation_generation_manifest.json'))
    p.add_argument('--runtime-root',type=Path,default=Path('.runtime/current/verification'))
    p.add_argument('--output-manifest',type=Path,default=Path('artifacts/manifests/stage11_claim_verification_manifest.json'))
    return p.parse_args()


def read(path: Path) -> list[dict[str,Any]]:
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x]


def write_rows(path: Path, rows: list[dict[str,Any]]) -> None:
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8',newline='\n') as h:
        for row in rows: h.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
    os.replace(tmp,path)


def bound(manifest: dict[str,Any], suffix: str) -> Path:
    paths=[Path(p) for p in manifest['output_artifact_hashes'] if p.endswith(suffix)]
    if len(paths)!=1: raise ValueError(f'Expected one {suffix}')
    if sha256_file(paths[0])!=manifest['output_artifact_hashes'][str(paths[0])]: raise ValueError(f'Hash mismatch: {paths[0]}')
    return paths[0]


def call(client: OllamaClient, model: str, rendered: dict[str,Any], schema: dict[str,Any], validator, retry: dict[str,Any]) -> tuple[Any,Any,int,list[dict[str,str]]]:
    errors=[]
    for attempt in range(int(retry['max_attempts'])+int(retry['repair_attempts'])+1):
        prompt=rendered['user_prompt'] if not attempt else rendered['user_prompt']+'\n\n'+retry['retry_instruction']
        try:
            result=client.generate(model,prompt,system_prompt=rendered['system_prompt'],json_format=schema,token_limit=client.defaults['structured_token_limit']*(2**attempt),timeout_seconds=client.defaults['timeout_seconds']*(2**attempt))
            return validator(json.loads(result.text)),result,attempt,errors
        except Exception as error: errors.append({'error_type':type(error).__name__,'message':str(error)})
    return None,None,int(retry['max_attempts'])+int(retry['repair_attempts']),errors


def main() -> None:
    a=args(); config=yaml.safe_load(a.config.read_text(encoding='utf-8')); models=yaml.safe_load(a.models_config.read_text(encoding='utf-8')); registry=load_prompt_registry(a.prompts)
    s10=json.loads(a.stage10_manifest.read_text(encoding='utf-8')); s9=json.loads(a.stage9_manifest.read_text(encoding='utf-8')); select=json.loads(a.selection_manifest.read_text(encoding='utf-8'))
    extraction_path=bound(s10,'extractions.jsonl'); generation_path=bound(s9,'explanations.jsonl'); input_path=bound(select,'condition_inputs.jsonl')
    extractions=read(extraction_path); generations=read(generation_path); inputs=read(input_path)
    if len(extractions)!=2987 or any(r['status']!='complete' for r in extractions): raise ValueError('Stage 11 requires 2,987 complete frozen Stage 10 records.')
    gen={(r['case_id'],r['generator'],r['condition']):r for r in generations if r['status']=='success'}; packets={(r['calibration_case_id'],r['condition']):r for r in inputs}
    if len(gen)!=len(extractions): raise ValueError('Extraction/generation accepted matrices differ.')
    rules=pd.read_csv(config['paths']['knowledge_base']); known=set(rules['rule_id'].astype(str)); verifier=models['verifier']; client=OllamaClient(models['generation_defaults'])
    run_id='stage11-claim-verification-'+sha256_file(extraction_path)[:12]; run=a.runtime_root/run_id; run.mkdir(parents=True,exist_ok=False)
    raw_path=run/'raw_phi_attempts.jsonl'; verified=[]; raw=[]
    try:
      for index, extraction in enumerate(extractions,1):
        key=(extraction['case_id'],extraction['generator'],extraction['condition']); generation=gen[key]; packet=packets[(extraction['case_id'],extraction['condition'])]; claims=extraction['claims']; trace=packet['B_exact_stored_trace']['rules']; trace_ids={str(x['rule_id']) for x in trace}
        context=packet['A_common_context']; case={'target_category':packet['target_category'],'query_group':packet['B_exact_stored_trace']['query_group']}; candidates=full_kb_candidate_retrieval(rules,target_category=str(packet['target_category']),query_group=str(case['query_group']))
        candidate_projection=[{k:r[k] for k in ('rule_id','rule_text','applicable_query_categories','required_context','query_terms','candidate_terms','recommended_category')} for r in candidates]
        rendered=render_prompt(registry,'claim_verification',{'full_kb_rules_json':json.dumps(candidate_projection,ensure_ascii=False),'exact_trace_rules_json':json.dumps(trace,ensure_ascii=False),'common_reference_facts_json':json.dumps(context,ensure_ascii=False),'explanation':generation['explanation'],'claims_json':json.dumps(claims,ensure_ascii=False)})
        payload,result,retries,errors=call(client,verifier['model_id'],rendered,separated_entailment_schema(),lambda x:validate_separated_entailment(x,claims,full_kb_rule_ids={str(x['rule_id']) for x in candidates},exact_trace_rule_ids=trace_ids,common_reference_item_facts=context),registry['roles']['claim_verification']['retry'])
        raw.append({'phase':'entailment','case_id':key[0],'generator':key[1],'condition':key[2],'raw_response_text':result.text if result else None,'errors':errors,'prompt_provenance':prompt_manifest_fields(rendered)})
        if payload is None:
            verified.append({'case_id':key[0],'generator':key[1],'condition':key[2],'status':'terminal_failure','claims':[],'retry_count':retries,'retry_errors':errors}); continue
        occurrences=citation_occurrences(generation['explanation'],known_rule_ids=known,trace_rule_ids=trace_ids)
        if occurrences and all(x['valid_canonical_occurrence'] for x in occurrences):
            cited=render_prompt(registry,'citation_validation',{'exact_trace_rules_json':json.dumps(trace,ensure_ascii=False),'citation_occurrences_json':json.dumps(occurrences,ensure_ascii=False),'claims_json':json.dumps(claims,ensure_ascii=False)})
            citation_payload,cres,cretries,cerrors=call(client,verifier['model_id'],cited,citation_validation_schema(),lambda x:validate_citation_validation(x,claims,occurrence_diagnostics=occurrences),registry['roles']['citation_validation']['retry'])
            raw.append({'phase':'citation','case_id':key[0],'generator':key[1],'condition':key[2],'raw_response_text':cres.text if cres else None,'errors':cerrors,'prompt_provenance':prompt_manifest_fields(cited)})
            if citation_payload is None: citation_rows=[{'claim_id':c['claim_id'],'citation_present':True,'canonical_citation_format':True,'cited_rule_ids':[],'invalid_rule_ids':[],'citation_entails_claim':None,'brief_reason':'citation semantic terminal failure'} for c in claims]
            else: citation_rows=citation_payload['claims']
        else:
            citation_rows=[{'claim_id':c['claim_id'],'citation_present':bool(occurrences),'canonical_citation_format':None if not occurrences else False,'cited_rule_ids':[],'invalid_rule_ids':[],'citation_entails_claim':None,'brief_reason':'N/A: no valid canonical citation occurrence'} for c in claims]
        cite={x['claim_id']:x for x in citation_rows}; rows=[]
        for claim,detail in zip(claims,payload['claims'],strict=True):
            eligibility=common_reference_eligibility(claim,context)['eligible']; c=cite[claim['claim_id']]
            rows.append({'claim_id':claim['claim_id'],'claim_text':claim['claim_text'],'claim_type':claim['claim_type'],'full_kb':'supported' if detail['full_kb_entailment']=='supported' else 'not_supported','full_kb_rule_ids':detail['full_kb_rule_ids'],'full_kb_reason':detail['full_kb_reason'],'exact_trace':'supported' if detail['exact_trace_entailment']=='supported' else 'not_supported','exact_trace_rule_ids':detail['exact_trace_rule_ids'],'exact_trace_reason':detail['exact_trace_reason'],'common_reference':'supported' if eligibility and detail['common_reference_item_fact_support']=='supported' else ('not_supported' if eligibility else 'N/A'),'common_reference_eligible':eligibility,'common_reference_fields':detail['common_reference_fields'],'common_reference_reason':detail['common_reference_reason'],'citation':c,'four_way_diagnostics':{'full_kb':detail['full_kb_entailment'],'exact_trace':detail['exact_trace_entailment'],'common_reference':detail['common_reference_item_fact_support']}})
        verified.append({'case_id':key[0],'generator':key[1],'condition':key[2],'status':'complete','claims':rows,'retry_count':retries,'retry_errors':errors,'entailment_prompt_provenance':prompt_manifest_fields(rendered),'explanation_sha256':text_sha256(generation['explanation'])})
        if index%50==0: print(f'stage11 verification {index}/{len(extractions)}',flush=True)
    finally: client.unload(verifier['model_id'])
    write_rows(run/'verifications.jsonl',verified); write_rows(raw_path,raw)
    total=sum(len(x['claims']) for x in verified); failures=sum(x['status']!='complete' for x in verified)
    manifest={'schema_version':1,'stage':11,'stage_name':'v3_claim_verification','run_id':run_id,'timestamp_utc':utc_timestamp(),'status':'frozen_verification_complete' if not failures else 'frozen_verification_with_terminal_failures','input_artifact_hashes':{str(a.stage10_manifest):sha256_file(a.stage10_manifest),str(extraction_path):sha256_file(extraction_path),str(a.selection_manifest):sha256_file(a.selection_manifest),str(a.stage9_manifest):sha256_file(a.stage9_manifest)},'output_artifact_hashes':{str(run/'verifications.jsonl'):sha256_file(run/'verifications.jsonl'),str(raw_path):sha256_file(raw_path)},'model':verifier,'row_counts':{'extractions':len(extractions),'verified_claims':total},'failure_counts':{'verification_terminal_failures':failures},'safeguards':{'strict_v3_trace_and_consequent_entailment':True,'full_kb_is_candidate_retrieval_then_applicability':True,'common_reference_eligibility_deterministic':True,'citation_syntax_deterministic':True,'raw_phi_responses_retained':True,'bounded_retries':True}}
    write_new_json(run/'manifest.json',manifest); write_json(a.output_manifest,manifest); print(json.dumps({'run_id':run_id,'verified_claims':total,'terminal_failures':failures},indent=2))

if __name__=='__main__': main()
