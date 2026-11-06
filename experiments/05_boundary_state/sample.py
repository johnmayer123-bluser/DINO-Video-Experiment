"""Fixed six-case qualitative comparison: 32-step Euler samples, shared noise."""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image,ImageDraw
from diffusers import AutoencoderKLWan
from model import ROOT,ARMS,BoundaryModel,setup,write_json


def labeled(frame,label):
    tile=Image.new('RGB',(192,216),'white')
    tile.paste(Image.fromarray(frame),(0,24));ImageDraw.Draw(tile).text((4,5),label,fill='black')
    return tile


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--steps',type=int,default=32)
    args=parser.parse_args();setup()
    root=args.root;cache=root/'cache/ucf6_12fps_192'
    rows=json.loads((cache/'manifest.json').read_text())
    chosen={}
    for row in sorted(rows,key=lambda row:row['id']):
        if row['split']=='test':chosen.setdefault(row['video'].split('/')[0],row)
    selected=[chosen[k] for k in sorted(chosen)]
    run=root/f'runs/boundary_state_seed{args.seed}'
    out=run/'samples';out.mkdir(exist_ok=True)
    checkpoint=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
    reference=torch.load(root/f'runs/history_width_seed{args.seed}/final.pt',map_location='cpu',weights_only=False)
    models={k:BoundaryModel(k).cuda().eval() for k in ARMS}
    models['S8'].load_state_dict(reference['models']['D256'])
    for k in ARMS[1:]:models[k].load_state_dict(checkpoint['models'][k])
    del checkpoint,reference
    vae=AutoencoderKLWan.from_pretrained(root/'models/wan-vae',torch_dtype=torch.float32,local_files_only=True).cuda().eval().requires_grad_(False)
    mean=torch.tensor(vae.config.latents_mean,device='cuda').reshape(1,16,1,1,1)
    std=torch.tensor(vae.config.latents_std,device='cuda').reshape(1,16,1,1,1)
    def decode(z):
        result=vae.decode(z*std+mean).sample[0]
        return ((result.clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
    reports=[]
    for i,row in enumerate(selected):
        z=torch.load(cache/'vae'/f"{row['id']}.pt",map_location='cpu',weights_only=True)['latent'][None].cuda().float()
        history=z[:,:,1:5]
        initial=torch.randn(z[:,:,5:9].shape,generator=torch.Generator().manual_seed(70042+i)).cuda()
        generated={}
        for name,model in models.items():
            x=initial.clone()
            with torch.autocast('cuda',dtype=torch.bfloat16):boundary,state=model.conditions(history)
            for step in range(args.steps):
                t=torch.full((1,),step/args.steps,device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):v=model.denoiser(boundary,x,t,state)
                x=x+v.float()/args.steps
            generated[name]=decode(torch.cat((z[:,:,:5],x),2))
        generated['VAE reference']=decode(z)
        names=[*ARMS,'VAE reference']
        video=out/f'case{i+1:02d}.mp4'
        with av.open(str(video),'w') as container:
            stream=container.add_stream('libx264',rate=12);stream.width=960;stream.height=216;stream.pix_fmt='yuv420p'
            for t in range(33):
                canvas=Image.new('RGB',(960,216))
                for j,name in enumerate(names):canvas.paste(labeled(generated[name][t],f'{name} | {"A" if t<17 else "B"} f{t}'),(j*192,0))
                frame=av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')
                for packet in stream.encode(frame):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        sheet=Image.new('RGB',(960,1080))
        for j,name in enumerate(names):
            for k,t in enumerate((16,17,20,24,32)):sheet.paste(labeled(generated[name][t],f'{name} | f{t}'),(k*192,j*216))
        sheet.save(out/f'case{i+1:02d}.png')
        reports.append({'case':i+1,'id':row['id'],'video':row['video'],'noise_seed':70042+i})
        print(json.dumps(reports[-1]),flush=True)
    write_json(out/'manifest.json',{'seed':args.seed,'euler_steps':args.steps,'fps':12,'selection':'First sorted test window per class; selected without model scores',
        'decoder':'All groups use identical observed A prefix for VAE decoding; denoisers only see their permitted conditions.',
        'panels':names,'cases':reports})


if __name__=='__main__':main()
