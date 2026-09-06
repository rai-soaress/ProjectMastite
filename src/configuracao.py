from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - mensagem tratada em tempo de execucao
    yaml = None


CONFIG_PADRAO: dict[str, Any] = {
    "wikimedia": {
        "max_imagens": 50,
        "termos_pesquisa": [
            "cow udder",
            "dairy cow udder",
            "cattle udder",
            "Holstein cow udder",
            "cow milking",
            "vaca leiteira úbere",
            "úbere bovino",
        ],
        "user_agent": "ProjetoMastiteAcademico/1.0 (coleta de imagens livres para estudo)",
        "timeout": 20,
        "intervalo_requisicoes": 1.0,
        "max_tentativas": 3,
        "max_tamanho_arquivo_mb": 15,
        "licencas_permitidas": [
            "cc0",
            "public domain",
            "cc-by",
            "cc-by-sa",
        ],
    },
    "imagens": {
        "largura_minima": 300,
        "altura_minima": 300,
        "formatos_permitidos": ["JPEG", "PNG"],
        "extensoes_permitidas": [".jpg", ".jpeg", ".png"],
        "distancia_hash_perceptual": 5,
    },
    "anotacao": {
        "modelo": "IDEA-Research/grounding-dino-tiny",
        "prompts": ["cow udder", "udder", "dairy cow udder", "cow teats"],
        "confianca_minima": 0.30,
        "area_minima": 0.002,
        "area_maxima": 0.80,
    },
    "split": {
        "treino": 0.70,
        "validacao": 0.20,
        "teste": 0.10,
        "semente": 42,
    },
    "yolo": {
        "modelo": "yolov8n.pt",
        "epocas": 30,
        "imgsz": 640,
        "batch": 8,
        "device": "cpu",
        "pasta_resultados": "runs/ubere",
    },
    "caminhos": {
        "downloads": "dataset/downloads",
        "descartadas": "dataset/descartadas",
        "revisao": "dataset/revisao",
        "pre_anotadas": "dataset/pre_anotadas",
        "images": "dataset/images",
        "labels": "dataset/labels",
        "metadata_csv": "metadata/imagens.csv",
        "log": "metadata/pipeline.log",
        "dataset_yaml": "dataset.yaml",
    },
}


def mesclar_config(base: dict[str, Any], sobrescrita: dict[str, Any]) -> dict[str, Any]:
    resultado = copy.deepcopy(base)
    for chave, valor in sobrescrita.items():
        if isinstance(valor, dict) and isinstance(resultado.get(chave), dict):
            resultado[chave] = mesclar_config(resultado[chave], valor)
        else:
            resultado[chave] = valor
    return resultado


def carregar_config(caminho: str | Path = "config.yaml") -> dict[str, Any]:
    caminho = Path(caminho)
    if not caminho.exists():
        logging.warning("Arquivo de configuracao nao encontrado. Usando valores padrao.")
        return copy.deepcopy(CONFIG_PADRAO)
    if yaml is None:
        logging.warning("PyYAML nao esta instalado. Usando configuracao padrao.")
        return copy.deepcopy(CONFIG_PADRAO)
    try:
        with caminho.open("r", encoding="utf-8") as arquivo:
            dados = yaml.safe_load(arquivo) or {}
    except FileNotFoundError as erro:
        logging.error("Arquivo de configuracao nao encontrado: %s", erro)
        return copy.deepcopy(CONFIG_PADRAO)
    except PermissionError as erro:
        logging.critical("Sem permissao para ler config.yaml: %s", erro)
        raise
    except OSError as erro:
        logging.critical("Erro ao ler config.yaml: %s", erro)
        raise
    return mesclar_config(CONFIG_PADRAO, dados)
