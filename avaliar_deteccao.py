"""Mede deteccoes no teste usando o mesmo limiar da interface."""
import json
from pathlib import Path
from PIL import Image
from src.configuracao import carregar_config
from src.inferencia import modelo_ativo
from src.treinar_yolo import _importar_yolo

RAIZ = Path(__file__).resolve().parent


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0


def main():
    import torch
    torch.set_num_threads(4)
    config=carregar_config()
    modelo=_importar_yolo()(str(modelo_ativo()))
    detalhes=[]
    tp=fp=fn=0
    pasta=RAIZ/'runs/ubere/teste_interface_limiar'
    pasta.mkdir(parents=True,exist_ok=True)
    for imagem in sorted((RAIZ/'dataset/images/test').glob('*.jpg')):
        with Image.open(imagem) as im:
            w,h=im.size
        reais=[]
        for linha in (RAIZ/'dataset/labels/test'/imagem.with_suffix('.txt').name).read_text().splitlines():
            _,x,y,bw,bh=map(float,linha.split())
            reais.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        r=modelo.predict(str(imagem),conf=config['yolo']['confianca'],iou=config['yolo']['iou'],imgsz=config['yolo']['imgsz'],device='cpu',verbose=False)[0]
        previstas=r.boxes.xyxy.cpu().tolist()
        disponiveis=set(range(len(reais)))
        acertos=0
        for box in previstas:
            melhor=max(disponiveis,key=lambda i:iou(box,reais[i]),default=None)
            if melhor is not None and iou(box,reais[melhor])>=0.5:
                acertos+=1
                disponiveis.remove(melhor)
        tp+=acertos; fp+=len(previstas)-acertos; fn+=len(disponiveis)
        Image.fromarray(r.plot()[:,:,::-1]).save(pasta/imagem.name)
        detalhes.append({'imagem':imagem.name,'acertos':acertos,'falsos_positivos':len(previstas)-acertos,'nao_detectados':len(disponiveis),'caixas':previstas,'confiancas':r.boxes.conf.cpu().tolist()})
    resultado={'modelo':str(modelo_ativo()),'confianca':config['yolo']['confianca'],'iou_nms':config['yolo']['iou'],'iou_acerto':0.5,'imagens':len(detalhes),'verdadeiros_positivos':tp,'falsos_positivos':fp,'falsos_negativos':fn,'precisao':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0,'detalhes':detalhes}
    (RAIZ/'metadata/avaliacao_limiar_interface.json').write_text(json.dumps(resultado,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in resultado.items() if k!='detalhes'}))


if __name__=='__main__':
    main()
