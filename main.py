from __future__ import annotations

import logging

from src.anotar_automaticamente import pre_anotar_uberes
from src.baixar_imagens import baixar_imagens
from src.configuracao import carregar_config
from src.preparar_yolo import preparar_dataset_yolo
from src.treinar_yolo import testar_modelo, treinar_yolo
from src.utils import configurar_logging, criar_pastas
from src.validar_imagens import validar_e_remover_duplicadas


MENU = """
1 - Baixar imagens
2 - Validar e remover duplicadas
3 - Pre-anotar os uberes
4 - Preparar dataset para YOLO
5 - Treinar YOLO
6 - Testar modelo
7 - Executar preparacao completa
0 - Sair
"""


def executar_etapa(nome: str, funcao, *args):
    try:
        resultado = funcao(*args)
        print(f"\n{nome} concluida: {resultado}\n")
        return resultado
    except (RuntimeError, FileNotFoundError, PermissionError, OSError, ValueError) as erro:
        logging.error("Falha na etapa '%s': %s", nome, erro)
        print(f"\nFalha na etapa '{nome}'. Consulte metadata/pipeline.log.\n")
    except KeyboardInterrupt:
        logging.warning("Etapa '%s' interrompida pelo usuario.", nome)
        print("\nOperacao interrompida. Arquivos incompletos foram tratados quando possivel.\n")
    finally:
        logging.info("Menu finalizou chamada da etapa: %s", nome)
    return None


def main() -> None:
    config = carregar_config()
    criar_pastas(config)
    configurar_logging(config)
    logging.info("Programa iniciado.")

    while True:
        print(MENU)
        opcao = input("Escolha uma opcao: ").strip()
        if opcao == "1":
            executar_etapa("Baixar imagens", baixar_imagens, config)
        elif opcao == "2":
            executar_etapa("Validar e remover duplicadas", validar_e_remover_duplicadas, config)
        elif opcao == "3":
            executar_etapa("Pre-anotar os uberes", pre_anotar_uberes, config)
        elif opcao == "4":
            executar_etapa("Preparar dataset para YOLO", preparar_dataset_yolo, config)
        elif opcao == "5":
            confirmar = input("O treino pode demorar. Digite TREINAR para iniciar: ").strip()
            if confirmar == "TREINAR":
                executar_etapa("Treinar YOLO", treinar_yolo, config)
            else:
                print("Treinamento cancelado.")
        elif opcao == "6":
            modelo = input("Caminho relativo do best.pt: ").strip()
            imagem = input("Caminho relativo da imagem de teste: ").strip()
            executar_etapa("Testar modelo", testar_modelo, config, modelo, imagem)
        elif opcao == "7":
            executar_etapa("Baixar imagens", baixar_imagens, config)
            executar_etapa("Validar e remover duplicadas", validar_e_remover_duplicadas, config)
            print("A pre-anotacao baixa/carrega modelo sob demanda. Use a opcao 3 quando quiser continuar.")
        elif opcao == "0":
            logging.info("Programa encerrado pelo usuario.")
            print("Saindo.")
            break
        else:
            print("Opcao invalida.")


if __name__ == "__main__":
    main()
