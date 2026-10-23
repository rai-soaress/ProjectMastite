from __future__ import annotations

import base64
import binascii
import io
import json
import math
from datetime import datetime, timezone
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from .treinar_yolo import _importar_yolo
from .utils import resolver_caminho


def modelo_ativo(caminho: str = ''):
    if not caminho:
        registro = resolver_caminho('metadata/modelo_ativo.json')
        if not registro.is_file():
            raise RuntimeError('Nenhum modelo treinado disponivel. Conclua o treino primeiro.')
        caminho = json.loads(registro.read_text(encoding='utf-8'))['caminho']
    modelo = resolver_caminho(caminho)
    if not modelo.is_file() or modelo.suffix != '.pt':
        raise ValueError('Modelo treinado nao encontrado.')
    return modelo


def salvar_upload(conteudo: str):
    try:
        bruto = base64.b64decode(conteudo, validate=True)
    except (ValueError, binascii.Error) as erro:
        raise ValueError('Imagem enviada em formato invalido.') from erro
    if not bruto or len(bruto) > 15 * 1024 * 1024:
        raise ValueError('Envie uma imagem de ate 15 MB.')
    try:
        with Image.open(io.BytesIO(bruto)) as original:
            if original.format not in ('JPEG', 'PNG', 'WEBP'):
                raise ValueError('Envie uma imagem JPG, PNG ou WebP.')
            if original.width * original.height > 40_000_000:
                raise ValueError('A imagem excede 40 megapixels. Reduza sua resolucao.')
            imagem = ImageOps.exif_transpose(original).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as erro:
        raise ValueError('Nao foi possivel abrir a imagem enviada.') from erro
    destino = resolver_caminho('dataset/uploads') / (uuid4().hex + '.jpg')
    destino.parent.mkdir(parents=True, exist_ok=True)
    imagem.save(destino, quality=95)
    return destino


def detectar_imagem(config: dict, dados: dict) -> dict:
    pesos = modelo_ativo(str(dados.get('modelo', '')).strip())
    if dados.get('arquivo_base64'):
        caminho = salvar_upload(dados['arquivo_base64'])
    else:
        caminho = resolver_caminho(str(dados.get('imagem', '')).strip())
        if not caminho.is_file():
            raise ValueError('Selecione uma foto de vaca para analisar.')
    YOLO = _importar_yolo()
    import torch
    torch.set_num_threads(int(config['yolo'].get('threads', 4)))
    modelo = YOLO(str(pesos))
    if list(modelo.names.values()) != ['ubere']:
        raise ValueError('Selecione o modelo treinado para a classe ubere.')
    resultado = modelo.predict(source=str(caminho), imgsz=int(config['yolo']['imgsz']), device=config['yolo']['device'], conf=float(config['yolo'].get('confianca', 0.25)), iou=float(config['yolo'].get('iou', 0.45)), verbose=False)[0]
    deteccoes = [{'classe':'ubere','confianca':round(float(c),4),'caixa_xyxy':[round(float(v),1) for v in box]} for box,c in zip(resultado.boxes.xyxy.cpu().tolist(), resultado.boxes.conf.cpu().tolist())]
    imagem = Image.fromarray(resultado.plot()[:,:,::-1])
    imagem.thumbnail((1600,1600))
    buffer = io.BytesIO()
    imagem.save(buffer,format='JPEG',quality=90)
    destino = resolver_caminho('runs/ubere/deteccoes')
    destino.mkdir(parents=True,exist_ok=True)
    identificador = uuid4().hex
    saida = destino / f'{identificador}_resultado.jpg'
    saida.write_bytes(buffer.getvalue())
    arquivo_original = f'{identificador}_original.jpg'
    with Image.open(caminho) as original:
        with ImageOps.exif_transpose(original).convert('RGB') as imagem_original:
            imagem_original.save(destino / arquivo_original, format='JPEG', quality=95)
    arquivo_recorte = None
    if deteccoes:
        melhor = max(deteccoes, key=lambda deteccao: deteccao['confianca'])
        with Image.open(caminho) as original:
            with ImageOps.exif_transpose(original).convert('RGB') as imagem_original:
                largura, altura = imagem_original.size
                x1, y1, x2, y2 = melhor['caixa_xyxy']
                margem_x = (x2 - x1) * 0.05
                margem_y = (y2 - y1) * 0.05
                caixa = (
                    max(0, math.floor(x1 - margem_x)),
                    max(0, math.floor(y1 - margem_y)),
                    min(largura, math.ceil(x2 + margem_x)),
                    min(altura, math.ceil(y2 + margem_y)),
                )
                recorte = imagem_original.crop(caixa)
        arquivo_recorte = f'{identificador}_recorte.jpg'
        recorte.save(destino / arquivo_recorte, format='JPEG', quality=95)
    mensagem = ('1 úbere detectado.' if len(deteccoes) == 1 else f'{len(deteccoes)} úberes detectados.') if deteccoes else 'Nenhum úbere detectado com confiança suficiente nesta foto.'
    registro = {
        'id': identificador,
        'data': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'mensagem': mensagem,
        'deteccoes': deteccoes,
        'arquivo_original': arquivo_original,
        'arquivo_resultado': saida.name,
        'arquivo_recorte': arquivo_recorte,
    }
    (destino / f'{identificador}.json').write_text(
        json.dumps(registro, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return {'id':identificador, **registro, 'imagem_resultado':'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii'), 'modelo':str(pesos)}
