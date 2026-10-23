import base64
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from src.inferencia import detectar_imagem, salvar_upload, modelo_ativo
import web_app


class TensorFalso:
    def __init__(self, valores):
        self.valores = valores

    def cpu(self):
        return self

    def tolist(self):
        return self.valores


class ImagemFalsa:
    def __getitem__(self, indice):
        return self


class InferenciaTests(unittest.TestCase):
    def test_upload_valida_e_salva_imagem(self):
        buffer = io.BytesIO()
        Image.new('RGB', (120,80), 'blue').save(buffer,format='PNG')
        with tempfile.TemporaryDirectory() as pasta:
            with patch('src.inferencia.resolver_caminho', return_value=Path(pasta)):
                arquivo = salvar_upload(base64.b64encode(buffer.getvalue()).decode())
                with Image.open(arquivo) as imagem:
                    self.assertEqual(imagem.size, (120,80))
                self.assertEqual(arquivo.suffix, '.jpg')

    def test_rejeita_upload_invalido(self):
        for dado in ('???', base64.b64encode(b'nao e uma imagem').decode(), ''):
            with self.assertRaises(ValueError):
                salvar_upload(dado)

    def test_modelo_automatico_exige_treino(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            with patch('src.inferencia.resolver_caminho', side_effect=lambda p: raiz/p):
                with self.assertRaisesRegex(RuntimeError, 'Nenhum modelo treinado'):
                    modelo_ativo()
                (raiz/'metadata').mkdir()
                (raiz/'modelo.pt').write_bytes(b'placeholder de teste')
                (raiz/'metadata/modelo_ativo.json').write_text(json.dumps({'caminho':'modelo.pt'}))
                self.assertEqual(modelo_ativo(),raiz/'modelo.pt')

    def test_salva_recorte_da_deteccao_com_maior_confianca(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            imagem_origem = raiz / 'vaca.jpg'
            Image.new('RGB', (100, 100), 'white').save(imagem_origem)
            modelo = SimpleNamespace(
                names={0: 'ubere'},
                predict=lambda **kwargs: [SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=TensorFalso([[0, 0, 20, 20], [40, 40, 80, 80]]),
                        conf=TensorFalso([0.7, 0.95]),
                    ),
                    plot=lambda: ImagemFalsa(),
                )],
            )
            resolver = lambda caminho: Path(caminho) if Path(caminho).is_absolute() else raiz / caminho
            torch_falso = SimpleNamespace(set_num_threads=lambda quantidade: None)
            with patch('src.inferencia.modelo_ativo', return_value=raiz / 'modelo.pt'), \
                    patch('src.inferencia._importar_yolo', return_value=lambda caminho: modelo), \
                    patch('src.inferencia.resolver_caminho', side_effect=resolver), \
                    patch('src.inferencia.Image.fromarray', return_value=Image.new('RGB', (100, 100))), \
                    patch.dict(sys.modules, {'torch': torch_falso}):
                resultado = detectar_imagem(
                    {'yolo': {'threads': 1, 'imgsz': 512, 'device': 'cpu'}},
                    {'imagem': str(imagem_origem)},
                )

            self.assertIsNotNone(resultado['arquivo_recorte'])
            self.assertEqual(len(resultado['deteccoes']), 2)
            pasta_deteccoes = raiz / 'runs/ubere/deteccoes'
            with Image.open(pasta_deteccoes / resultado['arquivo_recorte']) as recorte:
                self.assertEqual(recorte.size, (44, 44))
            self.assertTrue((pasta_deteccoes / resultado['arquivo_original']).is_file())
            self.assertTrue((pasta_deteccoes / resultado['arquivo_resultado']).is_file())
            registro = json.loads((pasta_deteccoes / f"{resultado['id']}.json").read_text(encoding='utf-8'))
            self.assertEqual(registro['id'], resultado['id'])
            with patch('web_app.resolver_caminho', return_value=pasta_deteccoes):
                historico = web_app._ler_historico()
            self.assertEqual(len(historico), 1)
            self.assertEqual(historico[0]['arquivo_original'], resultado['arquivo_original'])

    def test_nao_inicia_duas_operacoes_simultaneas(self):
        espera = threading.Event()
        terminou = threading.Event()
        def tarefa(config):
            espera.wait(3)
            terminou.set()
            return {'mensagem':'ok'}
        with web_app.ESTADO_LOCK:
            web_app.ESTADO['rodando'] = False
        try:
            self.assertTrue(web_app._iniciar_etapa('teste', tarefa, {}))
            self.assertFalse(web_app._iniciar_etapa('outro', tarefa, {}))
        finally:
            espera.set()
            terminou.wait(3)


if __name__ == '__main__':
    unittest.main()
