"""Generate larger Stage 1 hierarchical synthetic dataset."""
import argparse, json, os, random

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out', default='data/stage1_hierarchical_synthetic')
    p.add_argument('--n-projects', type=int, default=200)
    args=p.parse_args(); os.makedirs(args.out, exist_ok=True)
    sections=['Authentication','Deployment','Billing','Storage','Monitoring']
    docs=[]; refs=[]; qs=[]
    for i in range(1,args.n_projects+1):
        proj=f'project_{i:04d}'; base=f'memory://project/{proj}'
        docs.append({'id':proj+'_root','uri':base,'title':proj,'text':'Sections: '+', '.join(sections),'summary':f'Docs for {proj}.','anchors':sections})
        refs.append({'id':i,'token':f'<REF_{i}>','uri':base,'summary':f'Docs for {proj}.','metadata':{'stage':1}})
        for sec in sections:
            value=f'{sec.lower()}_{i}_value'
            docs.append({'id':f'{proj}_{sec}','uri':f'{base}#{sec}','title':sec,'text':f'{sec} detail value is {value}.','summary':f'{sec} summary for {proj}.','anchors':[f'{sec}.details']})
            docs.append({'id':f'{proj}_{sec}_details','uri':f'{base}#{sec}.details','title':sec+' details','text':f'The exact {sec} value is {value}.','summary':'Exact fact.','anchors':[]})
            qs.append({'id':f'q_{i}_{sec}','prompt':f'What is the {sec} value for {proj}? <REF_{i}>','answer':value,'expected_ref_ids':[i],'expected_anchors':[sec,f'{sec}.details']})
    for name, rows in [('documents.jsonl',docs),('references.jsonl',refs),('questions.jsonl',qs)]:
        with open(os.path.join(args.out,name),'w',encoding='utf-8') as f:
            for r in rows: f.write(json.dumps(r)+"\n")
if __name__=='__main__': main()
