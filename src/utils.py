from __future__ import annotations

import csv
import hashlib
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CAMPOS_CSV = [
    "nome_arquivo",
    "url_direta",
    "pagina_original",
    "autor",
    "licenca",
    "termo_pesquisado",
    "largura",
    "altura",
    "data_download",
    "status_processamento",
]


def raiz_projeto() -> Path:
    return Path(__file__).resolve().parent.parent


def resolver_caminho(caminho: str | Path) -> Path:
    raiz = raiz_projeto().resolve()
    destino = (raiz / caminho).resolve()
    if raiz != destino and raiz not in destino.parents:
        raise ValueError(f"Caminho fora da pasta do projeto: {caminho}")
    return destino


def criar_pastas(config: dict) -> None:
    chaves = ["downloads", "descartadas", "revisao", "pre_anotadas", "images", "labels"]
    for chave in chaves:
        resolver_caminho(config["caminhos"][chave]).mkdir(parents=True, exist_ok=True)
    for base in ["images", "labels"]:
        pasta = resolver_caminho(config["caminhos"][base])
        for parte in ["train", "val", "test"]:
            (pasta / parte).mkdir(parents=True, exist_ok=True)
    resolver_caminho(config["caminhos"]["metadata_csv"]).parent.mkdir(parents=True, exist_ok=True)


def configurar_logging(config: dict) -> None:
    caminho_log = resolver_caminho(config["caminhos"]["log"])
    caminho_log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(caminho_log, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def agora_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def nome_seguro(texto: str, limite: int = 80) -> str:
    texto = re.sub(r"[^A-Za-z0-9._-]+", "_", texto).strip("._")
    return (texto[:limite] or "imagem").lower()


def hash_arquivo(caminho: Path, bloco: int = 1024 * 1024) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        while parte := arquivo.read(bloco):
            resumo.update(parte)
    return resumo.hexdigest()


def garantir_csv(caminho_csv: Path) -> None:
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    if not caminho_csv.exists():
        with caminho_csv.open("w", newline="", encoding="utf-8") as arquivo:
            csv.DictWriter(arquivo, fieldnames=CAMPOS_CSV).writeheader()


def ler_metadata(caminho_csv: Path) -> list[dict[str, str]]:
    garantir_csv(caminho_csv)
    try:
        with caminho_csv.open("r", newline="", encoding="utf-8") as arquivo:
            return list(csv.DictReader(arquivo))
    except csv.Error as erro:
        logging.error("Erro ao ler CSV de metadata: %s", erro)
        raise
    except OSError as erro:
        logging.error("Erro de sistema ao ler metadata: %s", erro)
        raise


def anexar_metadata(caminho_csv: Path, linhas: Iterable[dict[str, object]]) -> None:
    garantir_csv(caminho_csv)
    try:
        with caminho_csv.open("a", newline="", encoding="utf-8") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=CAMPOS_CSV)
            for linha in linhas:
                escritor.writerow({campo: linha.get(campo, "") for campo in CAMPOS_CSV})
    except csv.Error as erro:
        logging.error("Erro ao gravar CSV de metadata: %s", erro)
        raise
    except OSError as erro:
        logging.error("Erro de sistema ao gravar metadata: %s", erro)
        raise


def atualizar_status(caminho_csv: Path, nome_arquivo: str, status: str) -> None:
    linhas = ler_metadata(caminho_csv)
    alterou = False
    for linha in linhas:
        if linha.get("nome_arquivo") == nome_arquivo:
            linha["status_processamento"] = status
            alterou = True
    if not alterou:
        logging.warning("Metadata nao encontrada para atualizar status: %s", nome_arquivo)
        return
    with caminho_csv.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=CAMPOS_CSV)
        escritor.writeheader()
        escritor.writerows(linhas)


def mover_com_seguranca(origem: Path, destino_dir: Path, motivo: str) -> Path:
    destino_dir.mkdir(parents=True, exist_ok=True)
    destino = destino_dir / origem.name
    contador = 1
    while destino.exists():
        destino = destino_dir / f"{origem.stem}_{contador}{origem.suffix}"
        contador += 1
    logging.warning("Movendo %s para %s. Motivo: %s", origem, destino, motivo)
    try:
        shutil.move(str(origem), str(destino))
    except FileNotFoundError as erro:
        logging.error("Arquivo nao encontrado ao mover descarte: %s", erro)
        raise
    except PermissionError as erro:
        logging.error("Sem permissao para mover arquivo: %s", erro)
        raise
    except OSError as erro:
        logging.error("Erro ao mover arquivo: %s", erro)
        raise
    return destino
