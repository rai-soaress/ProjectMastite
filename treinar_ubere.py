"""Treina o dataset preparado e avalia separadamente as fotos de teste."""
import json
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(RAIZ/'.cache/ultralytics'))
os.environ.setdefault('OMP_NUM_THREADS', '4')

from src.configuracao import carregar_config
from src.treinar_yolo import treinar_yolo
from src.utils import configurar_logging


def main():
    from ultralytics import YOLO
    config = carregar_config(RAIZ/'config.yaml')
    configurar_logging(config)
    best = treinar_yolo(config)
    modelo = YOLO(str(best))
    metricas = modelo.val(data=str(RAIZ/'dataset.yaml'), split='test', imgsz=config['yolo']['imgsz'], batch=4, device='cpu', workers=0, project=str(RAIZ/'runs/ubere'), name='avaliacao_teste')
    resumo = {'modelo':str(best), 'imagens_teste':len(list((RAIZ/'dataset/images/test').glob('*.jpg'))), 'metricas':{k:float(v) for k,v in metricas.results_dict.items()}, 'observacao':'Dataset pequeno; teste separado do treino e validacao. Nao comprova desempenho em outras fazendas.'}
    (RAIZ/'metadata/avaliacao_yolo.json').write_text(json.dumps(resumo,indent=2),encoding='utf-8')
    modelo.predict(source=str(RAIZ/'dataset/images/test'),conf=0.25,imgsz=config['yolo']['imgsz'],device='cpu',save=True,project=str(RAIZ/'runs/ubere'),name='predicoes_teste')
    print(json.dumps(resumo),flush=True)


if __name__ == '__main__':
    main()
