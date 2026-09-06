from __future__ import annotations

import logging
from pathlib import Path

try:
    import imagehash
except ImportError:  # pragma: no cover - depende do ambiente do usuario
    imagehash = None

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except ImportError:  # pragma: no cover - depende do ambiente do usuario
    Image = None
    ImageOps = None
    UnidentifiedImageError = OSError

from .utils import criar_pastas, hash_arquivo, mover_com_seguranca, resolver_caminho, atualizar_status


def _salvar_normalizada(caminho: Path, imagem, formato: str) -> None:
    imagem = ImageOps.exif_transpose(imagem)
    if formato == "JPEG" and imagem.mode not in ("RGB", "L"):
        imagem = imagem.convert("RGB")
    imagem.save(caminho, format=formato, quality=92)


def validar_e_remover_duplicadas(config: dict) -> dict[str, int]:
    if Image is None or ImageOps is None or imagehash is None:
        raise RuntimeError("Pillow e ImageHash sao obrigatorios. Execute: pip install -r requirements.txt")
    criar_pastas(config)
    pasta_downloads = resolver_caminho(config["caminhos"]["downloads"])
    pasta_descartadas = resolver_caminho(config["caminhos"]["descartadas"])
    csv_metadata = resolver_caminho(config["caminhos"]["metadata_csv"])
    hashes_exatos: dict[str, Path] = {}
    hashes_visuais: list[tuple[imagehash.ImageHash, Path]] = []
    resumo = {"validas": 0, "repetidas": 0, "descartadas": 0, "erros": 0}

    for caminho in sorted(pasta_downloads.iterdir()):
        if not caminho.is_file() or caminho.suffix.lower() not in config["imagens"]["extensoes_permitidas"]:
            continue
        try:
            with Image.open(caminho) as imagem:
                formato = imagem.format or ""
                largura, altura = imagem.size
                if formato not in config["imagens"]["formatos_permitidos"]:
                    raise ValueError(f"Formato nao permitido: {formato}")
                if largura < config["imagens"]["largura_minima"] or altura < config["imagens"]["altura_minima"]:
                    raise ValueError(f"Imagem muito pequena: {largura}x{altura}")
                imagem.load()
                hash_visual = imagehash.phash(imagem)
                _salvar_normalizada(caminho, imagem, formato)

            hash_exato = hash_arquivo(caminho)
            if hash_exato in hashes_exatos:
                mover_com_seguranca(caminho, pasta_descartadas, "duplicata exata")
                atualizar_status(csv_metadata, caminho.name, "descartada_duplicata_exata")
                resumo["repetidas"] += 1
                continue
            distancia_max = int(config["imagens"]["distancia_hash_perceptual"])
            if any(abs(hash_visual - existente) <= distancia_max for existente, _ in hashes_visuais):
                mover_com_seguranca(caminho, pasta_descartadas, "duplicata visual")
                atualizar_status(csv_metadata, caminho.name, "descartada_duplicata_visual")
                resumo["repetidas"] += 1
                continue

            hashes_exatos[hash_exato] = caminho
            hashes_visuais.append((hash_visual, caminho))
            atualizar_status(csv_metadata, caminho.name, "validada")
            resumo["validas"] += 1
            logging.info("Imagem valida: %s", caminho.name)
        except (UnidentifiedImageError, ValueError, OSError) as erro:
            resumo["descartadas"] += 1
            logging.error("Imagem descartada %s: %s", caminho.name, erro)
            try:
                mover_com_seguranca(caminho, pasta_descartadas, str(erro))
                atualizar_status(csv_metadata, caminho.name, "descartada_invalida")
            except (OSError, PermissionError, FileNotFoundError) as mover_erro:
                resumo["erros"] += 1
                logging.error("Falha ao mover imagem descartada %s: %s", caminho.name, mover_erro)
        finally:
            logging.debug("Validacao finalizada para %s", caminho.name)

    logging.info("Resumo de validacao: %s", resumo)
    return resumo
