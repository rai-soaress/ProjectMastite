from __future__ import annotations

import logging
import inspect
from pathlib import Path

try:
    from PIL import Image, ImageDraw, UnidentifiedImageError
except ImportError:  # pragma: no cover - depende do ambiente do usuario
    Image = None
    ImageDraw = None
    UnidentifiedImageError = OSError

from .utils import criar_pastas, mover_com_seguranca, resolver_caminho


def _carregar_grounding_dino(config: dict):
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as erro:
        raise RuntimeError("Instale torch e transformers com: pip install -r requirements.txt") from erro

    modelo_id = config["anotacao"]["modelo"]
    try:
        processor = AutoProcessor.from_pretrained(modelo_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(modelo_id)
        device = "cuda" if torch.cuda.is_available() and config["yolo"].get("device") == "cuda" else "cpu"
        model.to(device)
        model.eval()
        return processor, model, torch, device
    except OSError as erro:
        logging.error("Erro ao carregar modelo Grounding DINO: %s", erro)
        raise RuntimeError("Nao foi possivel carregar o modelo de pre-anotacao.") from erro


def _caixa_valida(caixa: list[float], largura: int, altura: int, config: dict) -> bool:
    x1, y1, x2, y2 = caixa
    if x1 < 0 or y1 < 0 or x2 > largura or y2 > altura or x2 <= x1 or y2 <= y1:
        return False
    area_relativa = ((x2 - x1) * (y2 - y1)) / (largura * altura)
    return config["anotacao"]["area_minima"] <= area_relativa <= config["anotacao"]["area_maxima"]


def _converter_yolo(caixa: list[float], largura: int, altura: int) -> str:
    x1, y1, x2, y2 = caixa
    xc = ((x1 + x2) / 2) / largura
    yc = ((y1 + y2) / 2) / altura
    w = (x2 - x1) / largura
    h = (y2 - y1) / altura
    valores = [max(0.0, min(1.0, valor)) for valor in [xc, yc, w, h]]
    return "0 " + " ".join(f"{valor:.6f}" for valor in valores)


def pre_anotar_uberes(config: dict) -> dict[str, int]:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow e obrigatorio. Execute: pip install -r requirements.txt")
    criar_pastas(config)
    pasta_downloads = resolver_caminho(config["caminhos"]["downloads"])
    pasta_pre = resolver_caminho(config["caminhos"]["pre_anotadas"])
    pasta_revisao = resolver_caminho(config["caminhos"]["revisao"])
    pasta_labels = pasta_pre / "labels"
    pasta_debug = pasta_pre / "debug"
    pasta_labels.mkdir(parents=True, exist_ok=True)
    pasta_debug.mkdir(parents=True, exist_ok=True)

    processor, model, torch, device = _carregar_grounding_dino(config)
    parametros = inspect.signature(processor.post_process_grounded_object_detection).parameters
    chave_limiar = "box_threshold" if "box_threshold" in parametros else "threshold"
    resumo = {"anotadas": 0, "revisao": 0, "erros": 0}

    for caminho in sorted(pasta_downloads.iterdir()):
        if not caminho.is_file() or caminho.suffix.lower() not in config["imagens"]["extensoes_permitidas"]:
            continue
        try:
            imagem = Image.open(caminho).convert("RGB")
            largura, altura = imagem.size
            melhor: tuple[float, list[float]] | None = None
            for prompt in config["anotacao"]["prompts"]:
                inputs = processor(images=imagem, text=prompt.rstrip('. ') + '.', return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                resultados = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    **{chave_limiar: float(config["anotacao"]["confianca_minima"])},
                    text_threshold=0.25,
                    target_sizes=[(altura, largura)],
                )[0]
                for score, box in zip(resultados["scores"], resultados["boxes"]):
                    caixa = [float(v) for v in box.tolist()]
                    confianca = float(score)
                    if _caixa_valida(caixa, largura, altura, config) and (melhor is None or confianca > melhor[0]):
                        melhor = (confianca, caixa)

            if melhor is None:
                mover_com_seguranca(caminho, pasta_revisao, "sem deteccao de ubere")
                resumo["revisao"] += 1
                continue

            _, caixa = melhor
            label = _converter_yolo(caixa, largura, altura)
            (pasta_labels / f"{caminho.stem}.txt").write_text(label + "\n", encoding="utf-8")
            debug = imagem.copy()
            desenho = ImageDraw.Draw(debug)
            desenho.rectangle(caixa, outline="red", width=4)
            debug.save(pasta_debug / caminho.name)
            resumo["anotadas"] += 1
            logging.info("Pre-anotacao criada para %s", caminho.name)
        except (RuntimeError, UnidentifiedImageError, OSError, ValueError) as erro:
            resumo["erros"] += 1
            logging.error("Falha na pre-anotacao de %s: %s", caminho.name, erro)
        finally:
            logging.debug("Etapa de pre-anotacao finalizada para %s", caminho.name)

    logging.info("Resumo de pre-anotacao: %s", resumo)
    return resumo
