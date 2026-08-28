# Porównanie klasycznych i nowoczesnych metod uczenia maszynowego w klasyfikacji emocji na podstawie tekstu w języku polskim

Kod eksperymentów do pracy magisterskiej o powyższym tytule.

Zadanie: klasyfikacja **wieloetykietowa** (multi-label) na ośmiu emocjach podstawowych
Plutchika — radość, smutek, zaufanie, wstręt, strach, gniew, przeczuwanie, zdziwienie.
Metryka główna: F1-Macro.

## Korpusy

| Korpus | Język | Rozmiar | Rola |
|---|---|---|---|
| [`clarin-pl/twitteremo`](https://huggingface.co/datasets/clarin-pl/twitteremo) | polski (natywny) | 35 921 | główny zbiór treningowy |
| [`go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions) | angielski → polski (NLLB-200) | 43 410 | eksperyment cross-lingualny |
| [`clarin-knext/CLARIN-Emo`](https://huggingface.co/datasets/clarin-knext/CLARIN-Emo) | polski (natywny) | 6 367 | ewaluacja cross-domenowa |

Podział 80/16/4 ze stratyfikacją wieloetykietową (`MultilabelStratifiedShuffleSplit`,
`random_state=42`), poza CLARIN-Emo, gdzie zachowano oficjalne splity PolEval 2024.

## Struktura

```
notebooks/     przygotowanie danych, EDA, diagnostyka szumu etykiet
experiments/   eksperymenty 03–41 (notatniki + skrypty)
kaggle/        kernele uruchamiane na GPU w chmurze (duże enkodery, LLM-y)
```

Numeracja plików w `experiments/` odpowiada kolejności eksperymentów w pracy.
`experiments/thesis_lib.py` zbiera funkcje wspólne (metryki, strojenie progów,
bootstrap CI); kernele w `kaggle/` są celowo samowystarczalne, bo działają
w odizolowanym środowisku.

## Zakres eksperymentów

- reprezentacje: BoW, TF-IDF (word i char n-gram), fastText, LSA, LDA, statystyki
  tekstu, cechy leksykonowe NRC EmoLex
- modele klasyczne: LogReg, LinearSVC, Ridge, ComplementNB, RF, ExtraTrees,
  HistGB, LightGBM, MLP, KNN
- strategie wieloetykietowe: Binary Relevance, ClassifierChain, strojenie progów
  per etykieta, stacking
- transformery: HerBERT (base/large), XLM-RoBERTa (base/large), pełny fine-tuning
  i LoRA
- LLM-y: Bielik (QLoRA z głowicą klasyfikacyjną oraz promptowanie zero-shot/few-shot)
- analizy dodatkowe: krzywe uczenia, macierze transferu 3×3, augmentacja klas
  rzadkich, korekta tekstu, pooling korpusów, ensemble, testy sparowane

## Uruchamianie

Zależności w `pyproject.toml` (`uv sync`). Wymagany model spaCy `pl_core_news_lg`
oraz `HF_TOKEN` w `.env` dla modeli z ograniczonym dostępem.

Repozytorium zawiera **wyłącznie kod**. Korpusy pobierane są z HuggingFace przez
`notebooks/01_data_preparation.ipynb`, który generuje `data/processed/`. Leksykon
NRC EmoLex wymaga osobnego pobrania z
[saifmohammad.com](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
