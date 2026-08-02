"""Generate larger Stage 0 flat synthetic memory QA dataset."""
import argparse, json, os, random

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out', default='data/stage0_synthetic_memory')
    p.add_argument('--n', type=int, default=1000)
    args=p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    colors=['red','blue','green','yellow','purple','orange']
    shapes=['cube','sphere','pyramid','cylinder','cone']
    docs=[]; refs=[]; qs=[]
    for i in range(1,args.n+1):
        color=random.choice(colors); shape=random.choice(shapes); code=f'CODE{i:05d}'
        uri=f'memory://object/{i}'
        text=f'Object {i} is a {color} {shape}. Its secret code is {code}.'
        docs.append({'id':f'obj{i}','uri':uri,'title':f'Object {i}','text':text,'summary':f'Object {i} summary.','anchors':[]})
        refs.append({'id':i,'token':f'<REF_{i}>','uri':uri,'summary':f'Object {i} summary.','metadata':{'stage':0}})
        qs.append({'id':f'q_color_{i}','prompt':f'What color is object {i}? <REF_{i}>','answer':color,'expected_ref_ids':[i],'expected_anchors':[]})
        qs.append({'id':f'q_shape_{i}','prompt':f'What shape is object {i}? <REF_{i}>','answer':shape,'expected_ref_ids':[i],'expected_anchors':[]})
        qs.append({'id':f'q_code_{i}','prompt':f'What is the secret code of object {i}? <REF_{i}>','answer':code,'expected_ref_ids':[i],'expected_anchors':[]})
    for name, rows in [('documents.jsonl',docs),('references.jsonl',refs),('questions.jsonl',qs)]:
        with open(os.path.join(args.out,name),'w',encoding='utf-8') as f:
            for r in rows: f.write(json.dumps(r)+"\n")
if __name__=='__main__': main()
