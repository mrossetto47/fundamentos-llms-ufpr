# Fundamentos de LLMs — UFPR

Projeto desenvolvido na disciplina **Fundamentos de LLMs**, da Universidade Federal do Paraná (UFPR), para investigar a adaptação de um modelo de linguagem pré-treinado a uma tarefa de **classificação de sentimentos**.

## Objetivo

Substituir a cabeça de modelagem de linguagem (*language modeling head*) por uma cabeça de classificação binária e avaliar seu desempenho. A primeira etapa utiliza **transfer learning**: os parâmetros do Transformer são congelados e somente a nova cabeça de classificação é treinada.

O projeto também prevê, como extensões, investigar outras configurações de treinamento e comparar a classificação com abordagens baseadas em *prompts*. Essas extensões **ainda não foram executadas**.

## Experimento implementado

- **Backbone:** `meta-llama/Llama-3.2-1B`.
- **Arquitetura:** `AutoModelForSequenceClassification` com duas classes: negativa (0) e positiva (1).
- **Dataset:** [Stanford Sentiment Treebank 2 (SST-2)](https://huggingface.co/datasets/stanfordnlp/sst2), acessado via Hugging Face Datasets.
- **Treinamento:** subconjunto de 1.000 exemplos, embaralhados com `seed=42`.
- **Validação:** conjunto de validação do SST-2, com 872 exemplos.
- **Tokenizer:** comprimento máximo de 128 tokens; padding com o token EOS.
- **Parâmetros atualizados:** apenas a cabeça linear, com 4.096 parâmetros; backbone congelado.
- **Otimizador:** AdamW, `learning_rate=2e-5`, `weight_decay=0.01`.
- **Batch size:** 8; **épocas:** 3.
- **Precisão:** backbone em FP16; cabeça em FP32; `GradScaler` iniciado em 128.
- **Ambiente experimental:** Google Colab com GPU NVIDIA Tesla T4.

## Resultados observados

| Época | Loss de treinamento | Acurácia de treinamento | Loss de validação | Acurácia de validação |
|---|---:|---:|---:|---:|
| 1 | 0,863192 | 44,00% | 0,766756 | 50,80% |
| 2 | 0,805897 | 46,70% | 0,717696 | 53,10% |
| 3 | 0,756613 | 52,70% | 0,657471 | **61,12%** |

A **baseline majoritária** do conjunto de validação foi **50,92%**, correspondente a prever sempre a classe positiva. Após três épocas, a acurácia de validação foi 61,12%.

Matriz de confusão da terceira época (linhas: classe real; colunas: classe prevista):

```text
                  Previsto 0   Previsto 1
Real 0                295          133
Real 1                206          238
```

O classificador melhorou sua capacidade de identificar a classe negativa ao longo das épocas, mas ainda apresenta erros relevantes em ambas as classes. Os resultados correspondem a **uma execução experimental**, não a uma estimativa com múltiplas sementes ou a uma avaliação independente de teste.

## Como executar

1. Abra [`notebooks/01_transfer_learning_sst2.ipynb`](notebooks/01_transfer_learning_sst2.ipynb) no Google Colab.
2. Ative uma GPU em **Ambiente de execução → Alterar tipo de ambiente de execução**.
3. Instale as dependências de `requirements.txt` caso não estejam disponíveis no ambiente. No Colab, a célula de instalação do notebook pode ser habilitada.
4. Solicite acesso ao modelo no [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B), quando necessário, e utilize um token pessoal com permissão adequada. **Não publique tokens no código.**
5. Execute as células do notebook na ordem apresentada. O treinamento pode depender da memória disponível e das versões das bibliotecas.

O notebook documenta também um diagnóstico de instabilidade numérica: no teste inicial, a escala padrão do `GradScaler` produziu gradientes não finitos. A escala inicial de 128 permitiu completar as três épocas sem passos pulados por overflow na execução registrada.

## Organização

```text
fundamentos-llms-ufpr/
├── notebooks/
│   └── 01_transfer_learning_sst2.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

Os checkpoints (`.pt`, `.pth`, `.ckpt` etc.) não são necessários no repositório e devem permanecer fora do controle de versão. Os pesos do backbone não são distribuídos neste projeto.

## Limitações e próximos experimentos

Este primeiro resultado usa apenas 1.000 exemplos de treinamento e mantém o backbone congelado. As comparações com outras taxas de aprendizado, outras estratégias de treinamento e inferência por *prompts* são trabalhos futuros. Como os resultados de validação orientaram decisões ao longo do desenvolvimento, uma comparação final deve reservar uma avaliação independente e explicitar as condições experimentais.

## Referências

- [Meta Llama 3.2 1B — Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [SST-2 — Hugging Face Datasets](https://huggingface.co/datasets/stanfordnlp/sst2)
- [Documentação do Transformers](https://huggingface.co/docs/transformers/index)
- [Documentação do PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)
