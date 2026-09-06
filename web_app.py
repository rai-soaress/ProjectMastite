from __future__ import annotations

import csv
import html
import json
import logging
import socket
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.anotar_automaticamente import pre_anotar_uberes
from src.baixar_imagens import baixar_imagens
from src.configuracao import carregar_config
from src.preparar_yolo import preparar_dataset_yolo
from src.treinar_yolo import testar_modelo, treinar_yolo
from src.utils import configurar_logging, criar_pastas, resolver_caminho
from src.validar_imagens import validar_e_remover_duplicadas


ESTADO = {
    "rodando": False,
    "etapa": "Nenhuma etapa em execucao",
    "resultado": None,
    "erro": None,
}
ESTADO_LOCK = threading.Lock()

ETAPAS = {
    "baixar": ("Baixar imagens", baixar_imagens),
    "validar": ("Validar e remover duplicadas", validar_e_remover_duplicadas),
    "anotar": ("Pre-anotar uberes", pre_anotar_uberes),
    "preparar": ("Preparar YOLO", preparar_dataset_yolo),
    "treinar": ("Treinar YOLO", treinar_yolo),
}


HTML = r"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Projeto Mastite - Pipeline</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --text: #20242a;
      --muted: #667085;
      --line: #d9dee7;
      --panel: #ffffff;
      --accent: #13795b;
      --accent-dark: #0f6049;
      --warn: #9a5b00;
      --error: #a32020;
      --shadow: 0 10px 26px rgba(23, 31, 44, .08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    header {
      background: #20303a;
      color: white;
      padding: 18px 22px;
      border-bottom: 4px solid #86b049;
    }
    header h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    header p {
      margin: 6px 0 0;
      color: #d9e2e8;
      font-size: 14px;
    }
    main {
      width: min(1180px, calc(100% - 32px));
      margin: 22px auto;
      display: grid;
      grid-template-columns: 330px 1fr;
      gap: 18px;
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    aside { padding: 16px; }
    section { overflow: hidden; }
    h2 {
      margin: 0 0 12px;
      font-size: 17px;
      letter-spacing: 0;
    }
    .actions {
      display: grid;
      gap: 8px;
    }
    button {
      width: 100%;
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      border-radius: 6px;
      min-height: 42px;
      font-size: 14px;
      cursor: pointer;
      text-align: left;
      padding: 0 12px;
    }
    button:hover { background: var(--accent-dark); }
    button.secondary {
      background: white;
      color: var(--accent-dark);
    }
    button.secondary:hover { background: #eef7f3; }
    button:disabled {
      opacity: .55;
      cursor: wait;
    }
    .status {
      display: grid;
      gap: 10px;
      margin-bottom: 16px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      border-radius: 999px;
      padding: 0 10px;
      background: #edf3f1;
      color: #124f3d;
      font-weight: 700;
      font-size: 13px;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .content-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 78px;
      background: #fff;
    }
    .metric strong {
      display: block;
      font-size: 24px;
      margin-bottom: 4px;
    }
    .metric span {
      color: var(--muted);
      font-size: 13px;
    }
    .panel-body {
      display: grid;
      grid-template-columns: 1fr 1fr;
      min-height: 360px;
    }
    .box {
      padding: 16px 18px;
      border-right: 1px solid var(--line);
      min-width: 0;
    }
    .box:last-child { border-right: 0; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
      color: #25313d;
      background: #f5f7fa;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      max-height: 390px;
      overflow: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 700; }
    input {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      margin-bottom: 8px;
    }
    .danger { color: var(--error); }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .grid { grid-template-columns: repeat(2, 1fr); }
      .panel-body { grid-template-columns: 1fr; }
      .box { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <header>
    <h1>Projeto Mastite</h1>
    <p>Pipeline para localizar uberes e preparar dataset YOLO. Nao realiza diagnostico de mastite.</p>
  </header>
  <main>
    <aside>
      <h2>Execucao</h2>
      <div class="status">
        <div id="running" class="pill">Carregando</div>
        <div id="stage" class="meta">Etapa atual: -</div>
        <div id="result" class="meta">Resultado: -</div>
      </div>
      <div class="actions">
        <button data-step="baixar">Baixar imagens</button>
        <button data-step="validar">Validar e remover duplicadas</button>
        <button data-step="anotar">Pre-anotar uberes</button>
        <button data-step="preparar">Preparar YOLO</button>
        <button data-step="treinar" class="secondary">Treinar YOLO</button>
      </div>
      <h2 style="margin-top:18px">Teste do modelo</h2>
      <input id="modelo" placeholder="runs/ubere/.../weights/best.pt">
      <input id="imagem" placeholder="dataset/images/test/imagem.jpg">
      <button id="testar" class="secondary">Testar imagem</button>
      <p class="meta">O treino e a pre-anotacao podem baixar modelos apenas quando acionados.</p>
    </aside>
    <section>
      <div class="content-head">
        <h2>Resumo do dataset</h2>
        <button id="refresh" class="secondary" style="width:120px;text-align:center">Atualizar</button>
      </div>
      <div class="grid">
        <div class="metric"><strong id="m-downloads">0</strong><span>downloads</span></div>
        <div class="metric"><strong id="m-descartadas">0</strong><span>descartadas</span></div>
        <div class="metric"><strong id="m-labels">0</strong><span>labels finais</span></div>
        <div class="metric"><strong id="m-metadata">0</strong><span>linhas metadata</span></div>
      </div>
      <div class="panel-body">
        <div class="box">
          <h2>Log</h2>
          <pre id="logs">Carregando...</pre>
        </div>
        <div class="box">
          <h2>Metadata recente</h2>
          <div style="overflow:auto; max-height:390px">
            <table>
              <thead>
                <tr><th>arquivo</th><th>termo</th><th>status</th></tr>
              </thead>
              <tbody id="metadata"></tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  </main>
  <script>
    async function api(path, options) {
      const response = await fetch(path, options);
      const data = await response.json();
      if (!response.ok) throw new Error(data.erro || 'Falha na requisicao');
      return data;
    }
    function setBusy(isBusy) {
      document.querySelectorAll('button[data-step], #testar').forEach(btn => btn.disabled = isBusy);
    }
    async function refresh() {
      const data = await api('/api/status');
      document.getElementById('running').textContent = data.rodando ? 'Rodando' : 'Parado';
      document.getElementById('stage').textContent = 'Etapa atual: ' + data.etapa;
      document.getElementById('result').textContent = data.erro ? 'Erro: ' + data.erro : 'Resultado: ' + JSON.stringify(data.resultado);
      document.getElementById('result').className = data.erro ? 'meta danger' : 'meta';
      document.getElementById('logs').textContent = data.logs || 'Sem logs ainda.';
      document.getElementById('m-downloads').textContent = data.contadores.downloads;
      document.getElementById('m-descartadas').textContent = data.contadores.descartadas;
      document.getElementById('m-labels').textContent = data.contadores.labels;
      document.getElementById('m-metadata').textContent = data.contadores.metadata;
      setBusy(data.rodando);

      const linhas = await api('/api/metadata');
      const tbody = document.getElementById('metadata');
      tbody.innerHTML = linhas.map(item => (
        '<tr><td>' + item.nome_arquivo + '</td><td>' + item.termo_pesquisado + '</td><td>' + item.status_processamento + '</td></tr>'
      )).join('');
    }
    document.querySelectorAll('button[data-step]').forEach(btn => {
      btn.addEventListener('click', async () => {
        await api('/api/run', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({etapa: btn.dataset.step})
        });
        refresh();
      });
    });
    document.getElementById('testar').addEventListener('click', async () => {
      await api('/api/testar', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          modelo: document.getElementById('modelo').value,
          imagem: document.getElementById('imagem').value
        })
      });
      refresh();
    });
    document.getElementById('refresh').addEventListener('click', refresh);
    setInterval(refresh, 2500);
    refresh();
  </script>
</body>
</html>
"""


def _ler_json(handler: BaseHTTPRequestHandler) -> dict:
    tamanho = int(handler.headers.get("Content-Length") or 0)
    if tamanho <= 0:
        return {}
    bruto = handler.rfile.read(tamanho).decode("utf-8")
    try:
        return json.loads(bruto)
    except json.JSONDecodeError as erro:
        raise ValueError(f"JSON invalido: {erro}") from erro


def _contar_arquivos(config: dict) -> dict[str, int]:
    def contar(caminho: Path, padroes: tuple[str, ...]) -> int:
        if not caminho.exists():
            return 0
        return sum(1 for item in caminho.rglob("*") if item.is_file() and item.name != ".gitkeep" and item.suffix.lower() in padroes)

    metadata = resolver_caminho(config["caminhos"]["metadata_csv"])
    linhas_metadata = 0
    if metadata.exists():
        try:
            with metadata.open("r", encoding="utf-8", newline="") as arquivo:
                linhas_metadata = max(0, sum(1 for _ in csv.DictReader(arquivo)))
        except (csv.Error, OSError) as erro:
            logging.error("Falha ao contar metadata: %s", erro)
    return {
        "downloads": contar(resolver_caminho(config["caminhos"]["downloads"]), (".jpg", ".jpeg", ".png")),
        "descartadas": contar(resolver_caminho(config["caminhos"]["descartadas"]), (".jpg", ".jpeg", ".png")),
        "labels": contar(resolver_caminho(config["caminhos"]["labels"]), (".txt",)),
        "metadata": linhas_metadata,
    }


def _ler_logs(config: dict, limite: int = 80) -> str:
    caminho = resolver_caminho(config["caminhos"]["log"])
    if not caminho.exists():
        return ""
    try:
        linhas = caminho.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(linhas[-limite:])
    except OSError as erro:
        return f"Falha ao ler log: {erro}"


def _ler_metadata(config: dict, limite: int = 20) -> list[dict[str, str]]:
    caminho = resolver_caminho(config["caminhos"]["metadata_csv"])
    if not caminho.exists():
        return []
    try:
        with caminho.open("r", encoding="utf-8", newline="") as arquivo:
            linhas = list(csv.DictReader(arquivo))[-limite:]
    except (csv.Error, OSError) as erro:
        logging.error("Falha ao ler metadata no front: %s", erro)
        return []
    resultado = []
    for linha in reversed(linhas):
        resultado.append({
            "nome_arquivo": html.escape(linha.get("nome_arquivo", "")),
            "termo_pesquisado": html.escape(linha.get("termo_pesquisado", "")),
            "status_processamento": html.escape(linha.get("status_processamento", "")),
        })
    return resultado


def _executar_em_thread(nome: str, funcao, config: dict, *args) -> None:
    with ESTADO_LOCK:
        ESTADO.update({"rodando": True, "etapa": nome, "resultado": None, "erro": None})
    try:
        logging.info("Front iniciou etapa: %s", nome)
        resultado = funcao(config, *args) if args else funcao(config)
        with ESTADO_LOCK:
            ESTADO.update({"resultado": str(resultado), "erro": None})
        logging.info("Front concluiu etapa %s: %s", nome, resultado)
    except Exception as erro:  # captura central para manter o servidor vivo
        logging.error("Falha na etapa %s: %s", nome, erro)
        logging.debug("Detalhes da falha:\n%s", traceback.format_exc())
        with ESTADO_LOCK:
            ESTADO.update({"resultado": None, "erro": str(erro)})
    finally:
        with ESTADO_LOCK:
            ESTADO["rodando"] = False


class Handler(BaseHTTPRequestHandler):
    config: dict = {}

    def _json(self, dados: dict | list, status: int = 200) -> None:
        corpo = json.dumps(dados, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self) -> None:
        rota = urlparse(self.path).path
        if rota == "/":
            corpo = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)
            return
        if rota == "/api/status":
            with ESTADO_LOCK:
                estado = dict(ESTADO)
            estado["contadores"] = _contar_arquivos(self.config)
            estado["logs"] = _ler_logs(self.config)
            self._json(estado)
            return
        if rota == "/api/metadata":
            self._json(_ler_metadata(self.config))
            return
        self._json({"erro": "Rota nao encontrada"}, 404)

    def do_POST(self) -> None:
        rota = urlparse(self.path).path
        try:
            dados = _ler_json(self)
        except ValueError as erro:
            self._json({"erro": str(erro)}, 400)
            return
        with ESTADO_LOCK:
            if ESTADO["rodando"]:
                self._json({"erro": "Ja existe uma etapa em execucao."}, 409)
                return
        if rota == "/api/run":
            etapa = dados.get("etapa")
            if etapa not in ETAPAS:
                self._json({"erro": "Etapa invalida."}, 400)
                return
            nome, funcao = ETAPAS[etapa]
            thread = threading.Thread(target=_executar_em_thread, args=(nome, funcao, self.config), daemon=True)
            thread.start()
            self._json({"ok": True, "etapa": nome})
            return
        if rota == "/api/testar":
            modelo = str(dados.get("modelo", "")).strip()
            imagem = str(dados.get("imagem", "")).strip()
            if not modelo or not imagem:
                self._json({"erro": "Informe modelo e imagem."}, 400)
                return
            thread = threading.Thread(
                target=_executar_em_thread,
                args=("Testar modelo", testar_modelo, self.config, modelo, imagem),
                daemon=True,
            )
            thread.start()
            self._json({"ok": True, "etapa": "Testar modelo"})
            return
        self._json({"erro": "Rota nao encontrada"}, 404)

    def log_message(self, formato: str, *args) -> None:
        logging.info("HTTP: " + formato, *args)


def encontrar_porta(inicio: int = 8000, fim: int = 8100) -> int:
    for porta in range(inicio, fim + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as teste:
            try:
                teste.bind(("127.0.0.1", porta))
                return porta
            except OSError:
                continue
    raise RuntimeError("Nenhuma porta livre encontrada entre 8000 e 8100.")


def main() -> None:
    config = carregar_config()
    criar_pastas(config)
    configurar_logging(config)
    Handler.config = config
    porta = encontrar_porta()
    servidor = ThreadingHTTPServer(("127.0.0.1", porta), Handler)
    print(f"Front rodando em: http://127.0.0.1:{porta}")
    logging.info("Front iniciado em http://127.0.0.1:%s", porta)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nFront encerrado.")
    finally:
        servidor.server_close()
        logging.info("Front encerrado.")


if __name__ == "__main__":
    main()
