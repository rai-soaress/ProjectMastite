# Projeto Mastite - Dataset de uberes para YOLO

Este projeto localiza o ubere em fotos de vacas usando YOLO. Inclui busca de imagens
no Wikimedia Commons, importacao de um dataset publico, revisao de anotacoes,
treinamento e uma interface local para enviar uma foto e visualizar as deteccoes.

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

### Identificar o ubere em uma foto

Na secao **Identificar ubere**, selecione uma foto JPG, PNG ou WebP e clique em
**Identificar ubere na foto**. A aplicacao usa automaticamente o ultimo modelo
treinado e mostra a foto com as caixas e a confianca das deteccoes. Se nao houver
deteccao suficiente, informa isso sem inventar uma caixa. O resultado tambem
fica salvo em `runs/ubere/deteccoes`. Quando encontrar o ubere, a tela oferece
o botao **Salvar úbere recortado** para baixar o recorte da deteccao com maior
confianca. A secao **Imagens salvas** permite rever analises anteriores; a foto
original, a imagem anotada e o registro da deteccao permanecem em
`runs/ubere/deteccoes`, mesmo depois de fechar e abrir a aplicacao.

Para repetir o treinamento do dataset ja preparado e avaliar o conjunto de teste:

```powershell
.\.venv\Scripts\python.exe treinar_ubere.py
```

O modelo escolhido pela validacao fica em `runs/ubere/treino_ubere*/weights/best.pt`.
O registro `metadata/modelo_ativo.json` seleciona o modelo na interface. As metricas
de teste ficam em `metadata/avaliacao_yolo.json`. Para conferir acertos, falsos
positivos e omissoes com o mesmo limiar da interface, execute
`.\.venv\Scripts\python.exe avaliar_deteccao.py`; o relatorio fica em
`metadata/avaliacao_limiar_interface.json`.

### Dataset revisado

Foram baixadas 100 fotos do repositorio publico
[Mastitis_Cow_Disease_Detection](https://github.com/ameyachopade/Mastitis_Cow_Disease_Detection).
A fonte declara Apache-2.0; uma copia dos termos esta em `metadata/LICENSE`.
As fotos incluem materiais de diferentes origens e marcas de fornecedores;
a declaracao do repositorio nao verifica os direitos individuais dessas fotos.

A selecao visual reteve 53 fotos, somadas a uma foto do Wikimedia: 37 de treino,
10 de validacao e 7 de teste. As caixas foram corrigidas visualmente para a classe
unica `ubere`, sem aproveitar os rotulos de doenca da fonte. Copias e espelhos
visiveis foram excluidos e a selecao passou por comparacao de hash perceptual.
O manifesto com URLs e revisao da fonte esta em `metadata/publico_manifesto.json`;
as caixas revisadas estao em `metadata/revisao_dataset.json`.

Este e um dataset pequeno, composto principalmente de fotos proximas do ubere.
As metricas locais nao garantem desempenho em fotos distantes, outras fazendas,
angulos diferentes ou uberes encobertos. Para melhorar, adicione fotos variadas
com caixas revisadas e mantenha uma parte independente para teste.

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

O treino usa uma unica classe, `0: ubere`. As caixas devem delimitar o ubere,
nao o corpo inteiro da vaca. Antes de treinar, execute as etapas de download,
validacao, pre-anotacao e preparacao do dataset, nessa ordem. Sao necessarias
pelo menos tres imagens anotadas para criar conjuntos separados; esse minimo
apenas permite executar o treino e nao garante qualidade de deteccao.

Se faltarem bibliotecas no ambiente, instale usando o mesmo Python da aplicacao:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Antes de treinar, revise manualmente as caixas geradas em `dataset/pre_anotadas/debug`. A pre-anotacao pode errar, marcar regioes incorretas ou deixar imagens importantes para revisao.

Para iniciar o treino:

```powershell
python main.py
```

Escolha a opcao `5` e digite `TREINAR` quando solicitado. Ao final, o caminho do `best.pt` sera informado no terminal e registrado no log. Normalmente ele fica dentro de `runs/ubere/treino_ubere*/weights/best.pt`.

## Licencas e limites

O download usa a API publica do Wikimedia Commons, sem chave de API e sem scraping do Google Imagens. Mesmo assim, confira as licencas antes de publicar, redistribuir ou usar o dataset fora do contexto academico.

Este projeto separa duas tarefas: localizar visualmente o ubere e diagnosticar mastite. O codigo implementa apenas localizacao do ubere. Qualquer analise de mastite exigiria outro dataset, validacao tecnica e avaliacao profissional.
