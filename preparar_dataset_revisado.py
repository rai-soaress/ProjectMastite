"""Caixas revisadas visualmente; coordenadas na previa 320 x 240, sem alterar as fotos."""
import json
import shutil
from pathlib import Path

import imagehash
from PIL import Image, ImageDraw, ImageOps

from src.anotar_automaticamente import _converter_yolo
from src.configuracao import carregar_config
from src.preparar_yolo import preparar_dataset_yolo
from src.utils import anexar_metadata, agora_iso, ler_metadata

RAIZ = Path(__file__).resolve().parent
CAIXAS = {
    0: (65,10,277,211), 1: (48,0,267,214), 2: (80,20,203,137), 3: (80,42,182,202),
    5: (28,22,258,185), 6: (121,15,271,168), 7: (105,18,219,166), 8: (52,18,317,201),
    11: (88,20,232,193), 13: (85,5,319,216), 14: (52,19,295,202), 16: (80,21,305,175),
    20: (104,16,235,174), 21: (119,36,260,183), 25: (133,26,249,170), 27: (76,22,307,222),
    28: (96,4,318,195), 30: (12,0,216,166), 31: (36,17,318,219), 33: (119,19,276,206),
    34: (108,50,274,154), 35: (88,47,268,201), 36: (51,92,97,157), 37: (109,8,276,181),
    39: (91,58,263,208), 40: (220,119,273,159), 41: (182,71,289,187), 43: (65,17,211,180),
    44: (70,0,273,198), 49: (32,19,257,214), 50: (74,0,230,136), 52: (38,0,171,139),
    58: (104,39,205,132), 59: (100,43,195,154), 60: (141,26,279,155), 64: (69,12,206,193),
    67: (45,8,271,196), 72: (78,94,188,199), 74: (87,40,182,115), 75: (112,62,210,149),
    78: (94,72,198,177), 79: (61,108,104,152), 80: (89,68,244,167), 84: (111,26,235,144),
    86: (66,0,264,228), 88: (131,87,227,166), 90: (119,47,226,171), 92: (79,0,244,212),
    93: (135,42,224,138), 94: (61,89,166,202), 95: (48,14,132,115), 97: (109,33,207,141),
    98: (58,12,224,205),
}


def main():
    dados = json.loads((RAIZ/'metadata/publico_manifesto.json').read_text())['imagens']
    hashes, aceitas, rejeitadas = [], [], []
    csv = RAIZ/'metadata/imagens.csv'
    registrados = {x['nome_arquivo'] for x in ler_metadata(csv)}
    for indice,box in CAIXAS.items():
        item = dados[indice]
        im = Image.open(RAIZ/item['arquivo']).convert('RGB')
        h, espelho = imagehash.phash(im), imagehash.phash(ImageOps.mirror(im))
        if any(min(h-outro, espelho-outro) <= 8 for outro in hashes):
            rejeitadas.append(indice)
            continue
        hashes.append(h)
        w,altura = im.size
        caixa = [box[0]*w/320,box[1]*altura/240,box[2]*w/320,box[3]*altura/240]
        nome = item['nome']+'.jpg'
        shutil.copy2(RAIZ/item['arquivo'], RAIZ/'dataset/downloads'/nome)
        label = RAIZ/'dataset/pre_anotadas/labels'/(item['nome']+'.txt')
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(_converter_yolo(caixa,w,altura)+'\n')
        ImageDraw.Draw(im).rectangle(caixa,outline='red',width=max(2,w//200))
        im.save(RAIZ/'dataset/pre_anotadas/debug'/nome)
        if nome not in registrados:
            anexar_metadata(csv,[dict(nome_arquivo=nome,url_direta=item['origem'],pagina_original='https://github.com/ameyachopade/Mastitis_Cow_Disease_Detection',autor='ameyachopade; ver fonte original',licenca='Apache-2.0 declarada pelo repositorio',termo_pesquisado='dataset publico de uberes',largura=w,altura=altura,data_download=agora_iso(),status_processamento='caixa revisada visualmente')])
        aceitas.append({'indice':indice,'arquivo':nome,'caixa_xyxy':caixa,'tamanho':[w,altura]})
    resumo = preparar_dataset_yolo(carregar_config())
    relatorio = {'selecionadas':aceitas,'duplicatas_descartadas':rejeitadas,'split':resumo,'criterio':'Revisao visual e deduplicacao perceptual incluindo espelhos; uma unica classe ubere. Anotacoes aproximadas, sem revisao veterinaria.'}
    (RAIZ/'metadata/revisao_dataset.json').write_text(json.dumps(relatorio,indent=2),encoding='utf-8')
    print(json.dumps({'novas_imagens':len(aceitas),'duplicatas':rejeitadas,'split':resumo}))


if __name__ == '__main__':
    main()
