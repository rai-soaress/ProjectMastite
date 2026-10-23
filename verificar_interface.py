"""Teste real no Edge headless (requer playwright apenas para desenvolvimento)."""
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parent


def main():
    url = os.environ.get('APP_TEST_URL', 'http://127.0.0.1:8001')
    foto = sorted((RAIZ/'dataset/images/test').glob('*.jpg'))[0]
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width':1440,'height':1000})
        erros = []
        page.on('pageerror', lambda e: erros.append(str(e)))
        page.goto(url)
        page.locator('#testar').click()
        page.wait_for_function("document.getElementById('erro-teste').textContent.includes('Selecione')")
        page.locator('#arquivo').set_input_files(str(foto))
        page.locator('#testar').click()
        page.locator('#deteccao').wait_for(state='visible',timeout=90000)
        page.wait_for_function("document.getElementById('deteccao-imagem').naturalWidth > 0")
        texto = page.locator('#deteccao-texto').inner_text()
        page.screenshot(path=str(RAIZ/'metadata/interface_deteccao.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        page.screenshot(path=str(RAIZ/'metadata/interface_mobile.png'),full_page=True)
        assert not erros, erros
        dados = page.request.get(url+'/api/status').json()
        assert dados['resultado']['imagem_resultado'].startswith('data:image/jpeg;base64,')
        assert Path(dados['resultado']['arquivo_resultado']).is_file()
        relatorio = {'url':url,'foto':str(foto),'resultado':texto,'erros_javascript':erros,'modelo':dados['resultado']['modelo'],'deteccoes':dados['resultado']['deteccoes']}
        (RAIZ/'metadata/teste_interface.json').write_text(json.dumps(relatorio,indent=2),encoding='utf-8')
        print(json.dumps(relatorio),flush=True)
        browser.close()


if __name__ == '__main__':
    main()
