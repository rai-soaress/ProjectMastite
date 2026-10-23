import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.configuracao import CONFIG_PADRAO
from src.preparar_yolo import _validar_label, preparar_dataset_yolo
from src.treinar_yolo import _confirmar_dataset
from src.baixar_imagens import _licenca_permitida


class DatasetTests(unittest.TestCase):
    def test_licencas_com_espacos_e_hifens(self):
        permitidas = ['cc0', 'public domain', 'cc-by', 'cc-by-sa']
        for licenca in ('CC BY-SA 4.0', 'CC-BY 2.0', 'CC0', 'Public domain'):
            self.assertTrue(_licenca_permitida(licenca, permitidas))
        for licenca in ('CC BY-NC 4.0', 'CC BY-ND 4.0', 'GFDL', ''):
            self.assertFalse(_licenca_permitida(licenca, permitidas))

    def test_rejeita_coordenadas_nao_finitas(self):
        with tempfile.TemporaryDirectory() as pasta:
            label = Path(pasta) / 'label.txt'
            for valor in ('nan', 'inf', '-inf'):
                label.write_text(f'0 {valor} 0.5 0.2 0.2')
                self.assertFalse(_validar_label(label))
            label.write_text('0 0.5 0.5 0.2 0.2')
            self.assertTrue(_validar_label(label))

    def test_dataset_vazio_tem_mensagem_util(self):
        with tempfile.TemporaryDirectory() as pasta:
            with patch('src.treinar_yolo.resolver_caminho', return_value=Path(pasta)):
                with self.assertRaisesRegex(RuntimeError, 'Dataset nao preparado'):
                    _confirmar_dataset(CONFIG_PADRAO)
                (Path(pasta) / 'train').mkdir()
                with self.assertRaisesRegex(RuntimeError, 'Dataset vazio'):
                    _confirmar_dataset(CONFIG_PADRAO)

    def test_preparacao_gera_caminhos_existentes_e_preserva_dataset_insuficiente(self):
        from PIL import Image
        import yaml
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            config = copy.deepcopy(CONFIG_PADRAO)
            resolver = lambda valor: raiz / valor
            with patch('src.preparar_yolo.resolver_caminho', side_effect=resolver), patch('src.utils.resolver_caminho', side_effect=resolver):
                downloads = raiz / config['caminhos']['downloads']
                labels = raiz / config['caminhos']['pre_anotadas'] / 'labels'
                downloads.mkdir(parents=True)
                labels.mkdir(parents=True)
                for i in range(3):
                    Image.new('RGB', (100, 100)).save(downloads / f'{i}.jpg')
                    (labels / f'{i}.txt').write_text('0 0.5 0.5 0.2 0.2')
                self.assertEqual(preparar_dataset_yolo(config), {'train': 1, 'val': 1, 'test': 1})
                dados = yaml.safe_load((raiz / 'dataset.yaml').read_text())
                for parte in ('train', 'val', 'test'):
                    self.assertTrue(Path(dados[parte]).is_absolute())
                    self.assertEqual(len(list(Path(dados[parte]).glob('*.jpg'))), 1)
                (labels / '0.txt').unlink()
                with self.assertRaisesRegex(RuntimeError, 'pelo menos 3'):
                    preparar_dataset_yolo(config)
                self.assertEqual(len(list(Path(dados['train']).glob('*.jpg'))), 1)


if __name__ == '__main__':
    unittest.main()
