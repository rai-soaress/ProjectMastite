from __future__ import annotations

import logging
from pathlib import Path

from .utils import resolver_caminho


def _importar_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as erro:
        raise RuntimeError("Instale ultralytics com: pip install -r requirements.txt") from erro
    return YOLO


def _confirmar_dataset(config: dict) -> None:
    for parte in ["train", "val", "test"]:
        pasta_img = resolver_caminho(config["caminhos"]["images"]) / parte
        pasta_lbl = resolver_caminho(config["caminhos"]["labels"]) / parte
        imagens = [p for p in pasta_img.iterdir() if p.suffix.lower() in config["imagens"]["extensoes_permitidas"]]
        for imagem in imagens:
            label = pasta_lbl / f"{imagem.stem}.txt"
            if not label.exists():
                raise RuntimeError(f"Imagem sem label antes do treino: {imagem}")


def treinar_yolo(config: dict) -> Path:
    _confirmar_dataset(config)
    YOLO = _importar_yolo()
    dataset_yaml = resolver_caminho(config["caminhos"]["dataset_yaml"])
    try:
        modelo = YOLO(config["yolo"]["modelo"])
        resultado = modelo.train(
            data=str(dataset_yaml),
            epochs=int(config["yolo"]["epocas"]),
            imgsz=int(config["yolo"]["imgsz"]),
            batch=int(config["yolo"]["batch"]),
            device=config["yolo"]["device"],
            project=config["yolo"]["pasta_resultados"],
            name="treino_ubere",
        )
        save_dir = Path(resultado.save_dir)
        best = save_dir / "weights" / "best.pt"
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
