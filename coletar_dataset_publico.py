"""Importa fotos e caixas de uma revisao fixa do dataset publico, sem executar codigo externo."""
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageOps

RAIZ = Path(__file__).resolve().parent
REPO = 'ameyachopade/Mastitis_Cow_Disease_Detection'
REVISAO = '2233d3161858d93714856361723e1069b630b61a'


def main():
    revision = REVISAO
    base = f'https://raw.githubusercontent.com/{REPO}/{revision}/'
    destino = RAIZ / 'dataset/publico'
    destino.mkdir(parents=True, exist_ok=True)
    (RAIZ / 'metadata').mkdir(exist_ok=True)
    for remoto, local in [('Dataset/annotations.json', 'annotations.json'), ('LICENSE', 'LICENSE')]:
        arquivo = RAIZ / 'metadata' / local
        if not arquivo.exists():
            resposta = requests.get(base + remoto, timeout=45)
            resposta.raise_for_status()
            arquivo.write_bytes(resposta.content)
    annotations = json.loads((RAIZ / 'metadata/annotations.json').read_text())

    def baixar(item):
        nome = 'publico_' + hashlib.sha256(item['filepath'].encode()).hexdigest()[:12]
        path = destino / (nome + '.jpg')
        if not path.exists():
            r = requests.get(base + quote(item['filepath']), timeout=45)
            r.raise_for_status()
            if r.content.startswith(b'version https://git-lfs.github.com/spec/v1'):
                r = requests.get(f'https://media.githubusercontent.com/media/{REPO}/{revision}/' + quote(item['filepath']), timeout=60)
                r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert('RGB')
            im.save(path, quality=95)
        im = Image.open(path)
        item = dict(item, arquivo=path.relative_to(RAIZ).as_posix(), nome=nome, tamanho=list(im.size), origem=base+quote(item['filepath']))
        print(nome, flush=True)
        return item

    with ThreadPoolExecutor(max_workers=4) as pool:
        dados = list(pool.map(baixar, annotations))
    (RAIZ / 'metadata/publico_manifesto.json').write_text(json.dumps({'repo': REPO, 'revision': revision, 'licenca_declarada': 'Apache-2.0', 'imagens': dados}, indent=2), encoding='utf-8')
    for inicio in range(0, len(dados), 20):
        lote = dados[inicio:inicio+20]
        sheet = Image.new('RGB', (1200, 250*((len(lote)+4)//5)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i,item in enumerate(lote):
            im = Image.open(RAIZ/item['arquivo']).convert('RGB')
            x,y,w,h = item['bbox']
            ImageDraw.Draw(im).rectangle((x,y,x+w,y+h), outline='red', width=max(2,im.width//200))
            sheet.paste(ImageOps.contain(im,(235,220)), ((i%5)*240,(i//5)*250))
            draw.text(((i%5)*240,(i//5)*250+222),f"{inicio+i}: {item['nome']}",fill='black')
        sheet.save(RAIZ / f'metadata/revisao_publico_{inicio//20}.jpg')
    print('TOTAL',len(dados))


if __name__ == '__main__':
    main()
