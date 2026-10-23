from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from urllib.parse import quote

try:
    import requests
except ImportError:  # pragma: no cover - depende do ambiente do usuario
    requests = None

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover - depende do ambiente do usuario
    Image = None
    UnidentifiedImageError = OSError

from .utils import (
    agora_iso,
    anexar_metadata,
    criar_pastas,
    hash_arquivo,
    ler_metadata,
    nome_seguro,
    resolver_caminho,
)


API_COMMONS = "https://commons.wikimedia.org/w/api.php"
TIPOS_VALIDOS = {"image/jpeg": ".jpg", "image/png": ".png"}


class LimiteRequisicoesError(RuntimeError):
    pass


def _requisicao_com_retentativa(session: requests.Session, metodo: str, url: str, config: dict, **kwargs):
    if requests is None:
        raise RuntimeError("requests nao esta instalado. Execute: pip install -r requirements.txt")
    max_tentativas = int(config["wikimedia"]["max_tentativas"])
    espera = 1.0
    ultima_excecao: Exception | None = None
    for tentativa in range(1, max_tentativas + 1):
        try:
            resposta = session.request(metodo, url, timeout=config["wikimedia"]["timeout"], **kwargs)
            resposta.raise_for_status()
            return resposta
        except requests.Timeout as erro:
            ultima_excecao = erro
            logging.warning("Timeout na tentativa %s para %s", tentativa, url)
        except requests.ConnectionError as erro:
            ultima_excecao = erro
            logging.warning("Falha de conexao na tentativa %s para %s: %s", tentativa, url, erro)
        except requests.HTTPError as erro:
            ultima_excecao = erro
            logging.error("Erro HTTP na tentativa %s para %s: %s", tentativa, url, erro)
            if erro.response is not None and erro.response.status_code == 429:
                raise LimiteRequisicoesError("Wikimedia limitou os downloads (HTTP 429). Aguarde antes de tentar novamente; as imagens ja baixadas foram preservadas.") from erro
            if erro.response is not None and 400 <= erro.response.status_code < 500:
                break
        except requests.RequestException as erro:
            ultima_excecao = erro
            logging.error("Erro de requisicao na tentativa %s para %s: %s", tentativa, url, erro)
        finally:
            if tentativa < max_tentativas:
                time.sleep(espera)
                espera *= 2
    raise RuntimeError(f"Falha apos {max_tentativas} tentativa(s): {ultima_excecao}")


def _licenca_permitida(licenca: str, permitidas: list[str]) -> bool:
    def normalizar(texto: str) -> str:
        return re.sub(r"[\s-]+", " ", texto.strip().lower())

    texto = normalizar(licenca or "")
    return any(re.fullmatch(re.escape(normalizar(item)) + r"(?: \d+(?:\.\d+)*)?", texto) for item in permitidas)


def _buscar_imagens(session: requests.Session, termo: str, limite: int, config: dict) -> list[dict]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f'{termo} filetype:bitmap',
        "gsrnamespace": 6,
        "gsrlimit": min(limite, 50),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1280,
        "format": "json",
    }
    resposta = _requisicao_com_retentativa(session, "GET", API_COMMONS, config, params=params)
    dados = resposta.json()
    paginas = dados.get("query", {}).get("pages", {})
    return sorted(paginas.values(), key=lambda pagina: pagina.get("index", 0))


def _validar_imagem_temporaria(caminho_tmp: Path, config: dict) -> tuple[str, int, int]:
    if Image is None:
        raise RuntimeError("Pillow nao esta instalado. Execute: pip install -r requirements.txt")
    try:
        with Image.open(caminho_tmp) as imagem:
            imagem.verify()
        with Image.open(caminho_tmp) as imagem:
            formato = imagem.format or ""
            largura, altura = imagem.size
    except UnidentifiedImageError as erro:
        raise ValueError(f"Arquivo baixado nao e uma imagem reconhecida: {erro}") from erro
    except OSError as erro:
        raise ValueError(f"Imagem corrompida ou ilegivel: {erro}") from erro

    if formato not in config["imagens"]["formatos_permitidos"]:
        raise ValueError(f"Formato nao permitido: {formato}")
    if largura < config["imagens"]["largura_minima"] or altura < config["imagens"]["altura_minima"]:
        raise ValueError(f"Imagem muito pequena: {largura}x{altura}")
    return formato, largura, altura


def baixar_imagens(config: dict) -> dict[str, int]:
    if requests is None or Image is None:
        raise RuntimeError("Dependencias ausentes. Execute: pip install -r requirements.txt")
    criar_pastas(config)
    pasta_downloads = resolver_caminho(config["caminhos"]["downloads"])
    csv_metadata = resolver_caminho(config["caminhos"]["metadata_csv"])
    existentes = ler_metadata(csv_metadata)
    urls_registradas = {linha.get("url_direta") for linha in existentes}
    max_total = int(config["wikimedia"]["max_imagens"])
    baixadas = 0
    puladas = 0
    erros = 0
    session = requests.Session()
    session.headers.update({"User-Agent": config["wikimedia"]["user_agent"]})

    for termo in config["wikimedia"]["termos_pesquisa"]:
        if baixadas >= max_total:
            break
        try:
            paginas = _buscar_imagens(session, termo, max_total - baixadas, config)
        except LimiteRequisicoesError:
            raise
        except (RuntimeError, ValueError, requests.RequestException) as erro:
            logging.error("Falha ao pesquisar termo '%s': %s", termo, erro)
            erros += 1
            continue

        for pagina in paginas:
            if baixadas >= max_total:
                break
            info = (pagina.get("imageinfo") or [{}])[0]
            url = info.get("thumburl") or info.get("url", "")
            mime = info.get("mime", "")
            metadata = info.get("extmetadata", {})
            licenca = metadata.get("LicenseShortName", {}).get("value", "")
            autor = metadata.get("Artist", {}).get("value", "")
            pagina_original = metadata.get("ObjectURL", {}).get("value") or f"https://commons.wikimedia.org/wiki/{quote(pagina.get('title', ''))}"

            if not url or url in urls_registradas:
                puladas += 1
                continue
            if mime not in TIPOS_VALIDOS:
                logging.warning("Ignorando %s por Content-Type informado: %s", url, mime)
                puladas += 1
                continue
            if not _licenca_permitida(licenca, config["wikimedia"]["licencas_permitidas"]):
                logging.warning("Ignorando %s por licenca nao permitida: %s", url, licenca)
                puladas += 1
                continue

            base = nome_seguro(f"{termo}_{pagina.get('pageid', hash_arquivo.__name__)}")
            extensao = TIPOS_VALIDOS[mime]
            destino = pasta_downloads / f"{base}{extensao}"
            contador = 1
            while destino.exists():
                destino = pasta_downloads / f"{base}_{contador}{extensao}"
                contador += 1
            tmp = destino.with_suffix(destino.suffix + ".tmp")
            try:
                resposta = _requisicao_com_retentativa(session, "GET", url, config, stream=True)
                content_type = resposta.headers.get("Content-Type", "").split(";")[0].lower()
                if content_type not in TIPOS_VALIDOS:
                    raise ValueError(f"Content-Type real nao permitido: {content_type}")
                tamanho = int(resposta.headers.get("Content-Length") or 0)
                limite_bytes = int(config["wikimedia"]["max_tamanho_arquivo_mb"] * 1024 * 1024)
                if tamanho and tamanho > limite_bytes:
                    raise ValueError(f"Arquivo maior que o limite configurado: {tamanho} bytes")

                total = 0
                with tmp.open("wb") as arquivo:
                    for bloco in resposta.iter_content(chunk_size=1024 * 128):
                        if not bloco:
                            continue
                        total += len(bloco)
                        if total > limite_bytes:
                            raise ValueError("Download ultrapassou o limite maximo de tamanho")
                        arquivo.write(bloco)

                _, largura, altura = _validar_imagem_temporaria(tmp, config)
                tmp.replace(destino)
                anexar_metadata(csv_metadata, [{
                    "nome_arquivo": destino.name,
                    "url_direta": url,
                    "pagina_original": pagina_original,
                    "autor": autor,
                    "licenca": licenca,
                    "termo_pesquisado": termo,
                    "largura": largura,
                    "altura": altura,
                    "data_download": agora_iso(),
                    "status_processamento": "baixada",
                }])
                urls_registradas.add(url)
                baixadas += 1
                logging.info("Imagem baixada e validada: %s", destino.name)
                time.sleep(float(config["wikimedia"]["intervalo_requisicoes"]))
            except LimiteRequisicoesError:
                raise
            except (RuntimeError, requests.RequestException, ValueError, OSError) as erro:
                erros += 1
                logging.error("Falha ao baixar/processar %s: %s", url, erro)
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError as erro:
                    logging.warning("Nao foi possivel remover arquivo temporario %s: %s", tmp, erro)

    resumo = {"baixadas": baixadas, "puladas": puladas, "erros": erros}
    logging.info("Resumo do download: %s", resumo)
    return resumo
