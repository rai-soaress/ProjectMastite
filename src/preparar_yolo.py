from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - depende do ambiente do usuario
    Image = None

from .utils import criar_pastas, resolver_caminho


def _validar_label(caminho_label: Path) -> bool:
    try:
        linhas = [linha.strip() for linha in caminho_label.read_text(encoding="utf-8").splitlines() if linha.strip()]
    except (FileNotFoundError, PermissionError, OSError) as erro:
        logging.error("Erro ao ler label %s: %s", caminho_label, erro)
        return False
    if not linhas:
        return False
    for linha in linhas:
        partes = linha.split()
        if len(partes) != 5 or partes[0] != "0":
            return False
        try:
            valores = [float(v) for v in partes[1:]]
        except ValueError:
            return False
        if any(valor < 0 or valor > 1 for valor in valores):
            return False
        if valores[2] <= 0 or valores[3] <= 0:
            return False
    return True


def _limpar_destinos(config: dict) -> None:
    for base_chave in ["images", "labels"]:
        base = resolver_caminho(config["caminhos"][base_chave])
        for parte in ["train", "val", "test"]:
            destino = base / parte
            destino.mkdir(parents=True, exist_ok=True)
            for item in destino.iterdir():
                if item.is_file() and item.name != ".gitkeep":
                    item.unlink()


def _dividir_pares(pares: list[tuple[Path, Path]], config: dict) -> dict[str, list[tuple[Path, Path]]]:
    total = len(pares)
    if total == 1:
        return {"train": pares, "val": [], "test": []}
    if total == 2:
        return {"train": pares[:1], "val": pares[1:], "test": []}

    n_train = max(1, int(total * float(config["split"]["treino"])))
    n_val = max(1, int(total * float(config["split"]["validacao"])))
    if n_train + n_val >= total:
        n_val = 1
        n_train = total - 2
    return {
        "train": pares[:n_train],
        "val": pares[n_train:n_train + n_val],
        "test": pares[n_train + n_val:],
    }


def preparar_dataset_yolo(config: dict) -> dict[str, int]:
    if Image is None:
        raise RuntimeError("Pillow e obrigatorio. Execute: pip install -r requirements.txt")
    criar_pastas(config)
    pasta_downloads = resolver_caminho(config["caminhos"]["downloads"])
    pasta_labels_pre = resolver_caminho(config["caminhos"]["pre_anotadas"]) / "labels"
    pasta_images = resolver_caminho(config["caminhos"]["images"])
    pasta_labels = resolver_caminho(config["caminhos"]["labels"])
    dataset_yaml = resolver_caminho(config["caminhos"]["dataset_yaml"])
    pares: list[tuple[Path, Path]] = []

    for imagem in sorted(pasta_downloads.iterdir()):
        if not imagem.is_file() or imagem.suffix.lower() not in config["imagens"]["extensoes_permitidas"]:
            continue
        label = pasta_labels_pre / f"{imagem.stem}.txt"
        if not label.exists():
            logging.warning("Imagem sem anotacao correspondente: %s", imagem.name)
            continue
        if not _validar_label(label):
            logging.warning("Label invalida ignorada: %s", label.name)
            continue
        try:
            with Image.open(imagem) as img:
                img.verify()
            pares.append((imagem, label))
        except OSError as erro:
            logging.error("Imagem invalida ao preparar YOLO %s: %s", imagem.name, erro)

    if not pares:
        raise RuntimeError("Nenhuma imagem com anotacao valida foi encontrada para preparar o YOLO.")

    rng = random.Random(int(config["split"]["semente"]))
    rng.shuffle(pares)
    conjuntos = _dividir_pares(pares, config)

    _limpar_destinos(config)
    resumo = {"train": 0, "val": 0, "test": 0}
    try:
        for parte, itens in conjuntos.items():
            for imagem, label in itens:
                shutil.copy2(imagem, pasta_images / parte / imagem.name)
                shutil.copy2(label, pasta_labels / parte / label.name)
                resumo[parte] += 1
    except (FileNotFoundError, PermissionError, OSError) as erro:
        logging.critical("Erro ao copiar arquivos para YOLO: %s", erro)
        raise
    finally:
        logging.info("Copia para estrutura YOLO finalizada.")

    dataset_yaml.write_text(
        "train: dataset/images/train\n"
        "val: dataset/images/val\n"
        "test: dataset/images/test\n\n"
        "names:\n"
        "  0: ubere\n",
        encoding="utf-8",
    )
    logging.info("dataset.yaml gerado em %s", dataset_yaml)
    return resumo
