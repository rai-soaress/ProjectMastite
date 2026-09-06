# Projeto Mastite - Dataset de uberes para YOLO

Este projeto cria uma automacao em Python para pesquisar imagens gratuitas no Wikimedia Commons, baixar apenas imagens com licenca identificada, validar os arquivos, remover duplicatas, gerar pre-anotacoes de uberes e organizar o dataset no formato esperado pelo YOLO.

O sistema localiza o ubere na imagem. Ele nao diagnostica mastite, nao estima gravidade e nao substitui avaliacao veterinaria.

## Instalacao

No PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## Front local

Para acompanhar a execucao pelo navegador, rode:

```powershell
python web_app.py
```

O terminal vai mostrar um endereco parecido com:

```text
http://127.0.0.1:8000
```

Abra esse link no navegador. A pagina mostra status da etapa atual, contadores do dataset, ultimas linhas do log e registros recentes do `metadata/imagens.csv`.

No CMD:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## Menu

`1 - Baixar imagens`: pesquisa no Wikimedia Commons pelos termos configurados em `config.yaml`, baixa em blocos, valida o tipo do arquivo e registra cada imagem em `metadata/imagens.csv`.

`2 - Validar e remover duplicadas`: corrige orientacao EXIF, valida tamanho/formato/integridade e move duplicatas ou imagens invalidas para `dataset/descartadas`.

`3 - Pre-anotar os uberes`: usa Grounding DINO via `transformers` e `torch` para sugerir uma caixa do ubere. As anotacoes ficam em `dataset/pre_anotadas/labels` e as imagens com caixas desenhadas ficam em `dataset/pre_anotadas/debug`.

`4 - Preparar dataset para YOLO`: copia imagens e labels validas para `dataset/images/{train,val,test}` e `dataset/labels/{train,val,test}`, dividindo em 70%, 20% e 10%.

`5 - Treinar YOLO`: inicia o treinamento com `ultralytics` somente apos confirmacao. O modelo padrao e pequeno (`yolov8n.pt`) e adequado para CPU.

`6 - Testar modelo`: roda predicao em uma imagem nova usando um `best.pt` ja treinado.

`7 - Executar preparacao completa`: executa download e validacao. A pre-anotacao e o treino continuam manuais porque podem baixar modelos e demorar.

## Arquivos e pastas

As imagens baixadas ficam em `dataset/downloads`. Imagens rejeitadas ficam em `dataset/descartadas`. Imagens sem deteccao automatica ficam em `dataset/revisao`.

As anotacoes automaticas em formato YOLO ficam em `dataset/pre_anotadas/labels`. Depois da preparacao, o dataset final fica em `dataset/images` e `dataset/labels`.

O arquivo `metadata/imagens.csv` guarda nome do arquivo, URL direta, pagina original, autor, licenca, termo pesquisado, largura, altura, data do download e status do processamento. O log fica em `metadata/pipeline.log`.

## YOLO

Antes de treinar, revise manualmente as caixas geradas em `dataset/pre_anotadas/debug`. A pre-anotacao pode errar, marcar regioes incorretas ou deixar imagens importantes para revisao.

Para iniciar o treino:

```powershell
python main.py
```

Escolha a opcao `5` e digite `TREINAR` quando solicitado. Ao final, o caminho do `best.pt` sera informado no terminal e registrado no log. Normalmente ele fica dentro de `runs/ubere/treino_ubere*/weights/best.pt`.

## Licencas e limites

O download usa a API publica do Wikimedia Commons, sem chave de API e sem scraping do Google Imagens. Mesmo assim, confira as licencas antes de publicar, redistribuir ou usar o dataset fora do contexto academico.

Este projeto separa duas tarefas: localizar visualmente o ubere e diagnosticar mastite. O codigo implementa apenas localizacao do ubere. Qualquer analise de mastite exigiria outro dataset, validacao tecnica e avaliacao profissional.
