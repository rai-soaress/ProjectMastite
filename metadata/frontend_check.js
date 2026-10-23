
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
      document.getElementById('result').textContent = data.erro ? 'Erro: ' + data.erro : 'Resultado: ' + (data.resultado?.mensagem || JSON.stringify(data.resultado));
      const deteccao = data.resultado?.imagem_resultado;
      document.getElementById('deteccao').style.display = deteccao ? 'block' : 'none';
      if (deteccao) {
        document.getElementById('deteccao-imagem').src = deteccao;
        document.getElementById('deteccao-texto').textContent = data.resultado.mensagem + ' ' + data.resultado.deteccoes.map(d => Math.round(d.confianca * 100) + '% de confianca').join(', ');
      }
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
      const erro = document.getElementById('erro-teste');
      erro.textContent = '';
      try {
      const arquivo = document.getElementById('arquivo').files[0];
      if (arquivo && arquivo.size > 15 * 1024 * 1024) throw new Error('Envie uma foto de ate 15 MB.');
      if (!arquivo && !document.getElementById('imagem').value.trim()) throw new Error('Selecione uma foto primeiro.');
      setBusy(true);
      const conteudo = arquivo ? await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = () => reject(new Error('Nao foi possivel ler a foto.'));
        reader.readAsDataURL(arquivo);
      }) : '';
      await api('/api/testar', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          modelo: document.getElementById('modelo').value,
          imagem: document.getElementById('imagem').value,
          arquivo_base64: conteudo
        })
      });
      await refresh();
      } catch (e) { erro.textContent = e.message; setBusy(false); }
    });
    document.getElementById('refresh').addEventListener('click', refresh);
    setInterval(refresh, 2500);
    refresh();
  