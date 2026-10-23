from __future__ import annotations

import logging
import json
import os
from pathlib import Path

from .utils import resolver_caminho
from .preparar_yolo import _validar_label


def _importar_yolo():
    os.environ.setdefault('YOLO_CONFIG_DIR', str(resolver_caminho('.cache/ultralytics')))
    try:
        from ultralytics import YOLO
    except ImportError as erro:
        raise RuntimeError("Instale ultralytics com: pip install -r requirements.txt") from erro
    return YOLO


def _confirmar_dataset(config: dict) -> None:
    for parte in ["train", "val", "test"]:
        pasta_img = resolver_caminho(config["caminhos"]["images"]) / parte
        pasta_lbl = resolver_caminho(config["caminhos"]["labels"]) / parte
        if not pasta_img.is_dir() or not pasta_lbl.is_dir():
            raise RuntimeError("Dataset nao preparado. Execute Baixar imagens, Pre-anotar uberes e Preparar YOLO primeiro.")
        imagens = [p for p in pasta_img.iterdir() if p.suffix.lower() in config["imagens"]["extensoes_permitidas"]]
        if not imagens and parte in ("train", "val"):
            raise RuntimeError(f"Dataset vazio em {parte}. Baixe imagens, anote os uberes e execute Preparar YOLO antes de treinar.")
        for imagem in imagens:
            label = pasta_lbl / f"{imagem.stem}.txt"
            if not label.exists():
                raise RuntimeError(f"Imagem sem label antes do treino: {imagem}")
            if not _validar_label(label):
                raise RuntimeError(f"Anotacao invalida de ubere: {label}")


def treinar_yolo(config: dict) -> Path:
    _confirmar_dataset(config)
    YOLO = _importar_yolo()
    import torch
    torch.set_num_threads(int(config['yolo'].get('threads', 4)))
    dataset_yaml = resolver_caminho(config["caminhos"]["dataset_yaml"])
    try:
        modelo = YOLO(config["yolo"]["modelo"])
        resultado = modelo.train(
            data=str(dataset_yaml),
            epochs=int(config["yolo"]["epocas"]),
            imgsz=int(config["yolo"]["imgsz"]),
            batch=int(config["yolo"]["batch"]),
            device=config["yolo"]["device"],
            project=str(resolver_caminho(config["yolo"]["pasta_resultados"])),
            workers=0,
            seed=int(config["split"]["semente"]),
            name="treino_ubere",
            patience=int(config['yolo'].get('patience', 20)),
            optimizer=config['yolo'].get('optimizer', 'AdamW'),
            lr0=float(config['yolo'].get('lr0', 0.0005)),
            warmup_bias_lr=float(config['yolo'].get('lr0', 0.0005)),
            freeze=int(config['yolo'].get('freeze', 10)),
            mosaic=float(config['yolo'].get('mosaic', 0.5)),
            scale=0.3,
        )
        save_dir = Path(resultado.save_dir)
        best = save_dir / "weights" / "best.pt"
        if not best.is_file():
            raise RuntimeError('Treino terminou sem gerar best.pt.')
        resolver_caminho('metadata/modelo_ativo.json').write_text(json.dumps({'caminho':str(best.resolve()), 'classe':'ubere'}, indent=2), encoding='utf-8')
        logging.info("Treinamento finalizado. Melhor modelo: %s", best)
        return best
    except (RuntimeError, OSError, ValueError) as erro:
        logging.critical("Falha ao treinar YOLO: %s", erro)
        raise
    finally:
        logging.info("Etapa de treino encerrada.")


def testar_modelo(config: dict, caminho_modelo: str, caminho_imagem: str) -> Path:
    YOLO = _importar_yolo()
    modelo_path = resolver_caminho(caminho_modelo)
    imagem_path = resolver_caminho(caminho_imagem)
    if not modelo_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {modelo_path}")
    if not imagem_path.exists():
        raise FileNotFoundError(f"Imagem nao encontrada: {imagem_path}")
    try:
        modelo = YOLO(str(modelo_path))
        resultados = modelo.predict(
            source=str(imagem_path),
            imgsz=int(config["yolo"]["imgsz"]),
            device=config["yolo"]["device"],
            conf=float(config['yolo'].get('confianca', 0.25)),
            iou=float(config['yolo'].get('iou', 0.45)),
            save=True,
            project=config["yolo"]["pasta_resultados"],
            name="teste_ubere",
        )
        pasta = Path(resultados[0].save_dir)
        logging.info("Teste finalizado. Resultado salvo em: %s", pasta)
        return pasta
    except (RuntimeError, OSError, ValueError) as erro:
        logging.error("Falha ao testar modelo YOLO: %s", erro)
        raise
    finally:
        logging.info("Etapa de teste encerrada.")
