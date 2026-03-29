<div align="center">

# Dataset for Identification of Homophobia and Transphobia<br>for Telugu, Kannada, and Gujarati

<p>
  <a href="https://aclanthology.org/2024.lrec-main.393/">
    <img src="https://img.shields.io/badge/ACL%20Anthology-2024.lrec--main.393-blue?style=for-the-badge&logo=semanticscholar" alt="ACL Anthology">
  </a>
  <a href="https://aclanthology.org/2024.lrec-main.393.pdf">
    <img src="https://img.shields.io/badge/PDF-LREC--COLING%202024-red?style=for-the-badge&logo=adobeacrobatreader" alt="PDF">
  </a>
  <img src="https://img.shields.io/badge/Languages-Telugu%20%7C%20Kannada%20%7C%20Gujarati-orange?style=for-the-badge" alt="Languages">
  <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey?style=for-the-badge" alt="License">
</p>

**Prasanna Kumar Kumaresan¹ · Rahul Ponnusamy¹ · Dhruv Sharma² · Paul Buitelaar¹ · Bharathi Raja Chakravarthi¹**

¹ Data Science Institute, University of Galway, Ireland &nbsp;|&nbsp; ² Indian Institute of Technology (BHU), Varanasi

*(LREC-COLING 2024 · Pages 4404–4411 · Torino, Italia · May 2024)*

</div>

---

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Dataset](#-dataset)
  - [Dataset Statistics](#dataset-statistics)
- [Repository Structure](#-repository-structure)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

---

> 🔬 *Disclaimer: This work is for educational and research purposes only, intended to classify and understand hate speech in a structured research framework. Examples do not represent the views of the authors or their affiliated institutions.*

---

## 📌 Abstract

Users of social media platforms are negatively affected by the proliferation of hate or abusive content. There has been a rise in homophobic and transphobic content in recent years targeting LGBT+ individuals. The increasing levels of homophobia and transphobia online can make online platforms harmful and threatening for LGBT+ persons, potentially inhibiting equality, diversity, and inclusion.

This paper makes the following contributions:

| # | Contribution |
|---|---|
| 1️⃣ | **First datasets** for homophobia and transphobia detection in **Telugu, Kannada, and Gujarati** — collected from YouTube comments |
| 2️⃣ | **~10,000 expert-labeled comments** per language with comprehensive annotation guidelines |
| 3️⃣ | **Baseline experiments** using pre-trained transformer models (XLM-RoBERTa, IndicBERT, monolingual BERT variants) achieving **>90% macro F1** across all three languages |

---

## 🗂️ Dataset

### Dataset Statistics

| Labels | Telugu | Kannada | Gujarati |
|---|---|---|---|
| **Homophobia** | 4,119 | 3,949 | 3,275 |
| **Transphobia** | 3,823 | 4,058 | 2,894 |
| **None of the categories** | 4,987 | 6,369 | 5,430 |

> ⚠️ **Ethics Notice:** All data was collected from public YouTube comments. User IDs and personal identifying information have been removed. The datasets are released strictly for research purposes. This dataset was also used in the [LT-EDI-2024@EACL](https://codalab.lisn.upsaclay.fr/competitions/16056) [Paper](https://aclanthology.org/2024.ltedi-1.11/) shared task competition.

---

## 📂 Repository Structure

```
LREC-COLING-2024-Homo-Trans/
│
├── 📁 Code/                          # Model training scripts
│   ├── BertGuj.py                    # BERT model — Gujarati
│   ├── BertKan.py                    # BERT model — Kannada
│   ├── BertTel.py                    # BERT model — Telugu
│   ├── IndicBERTGuj.py               # IndicBERT — Gujarati
│   ├── IndicBERTKan.py               # IndicBERT — Kannada
│   ├── IndicBERTTel.py               # IndicBERT — Telugu
│   ├── xlmRoBERTa_Guj.py            # XLM-RoBERTa — Gujarati
│   ├── xlmRoBERTa_Kan.py            # XLM-RoBERTa — Kannada
│   ├── xlmRoBERTa_Tel.py            # XLM-RoBERTa — Telugu
│   ├── functions.py                  # Shared utilities & helpers
│   └── config.yaml                   # Hyperparameter configuration
│
├── 📁 Datasets/                      # Train / Dev / Test CSVs
│   ├── Gujarati_train.csv
│   ├── Gujarati_dev.csv
│   ├── Gujarati_test.csv
│   ├── Kannada_train.csv
│   ├── Kannada_dev.csv
│   ├── Kannada_test.csv
│   ├── Telugu_train.csv
│   ├── Telugu_dev.csv
│   └── Telugu_test.csv
│
└── README.md
```
---

## 📖 Citation

If you use this dataset or code in your research, please cite:

```bibtex
@inproceedings{kumaresan-etal-2024-dataset,
    title = "Dataset for Identification of Homophobia and Transphobia for {T}elugu, {K}annada, and {G}ujarati",
    author = "Kumaresan, Prasanna Kumar  and
      Ponnusamy, Rahul  and
      Sharma, Dhruv  and
      Buitelaar, Paul  and
      Chakravarthi, Bharathi Raja",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.393/",
    pages = "4404--4411",
    abstract = "Users of social media platforms are negatively affected by the proliferation of hate or abusive content. There has been a rise in homophobic and transphobic content in recent years targeting LGBT+ individuals. The increasing levels of homophobia and transphobia online can make online platforms harmful and threatening for LGBT+ persons, potentially inhibiting equality, diversity, and inclusion. We are introducing a new dataset for three languages, namely Telugu, Kannada, and Gujarati. Additionally, we have created an expert-labeled dataset to automatically identify homophobic and transphobic content within comments collected from YouTube. We provided comprehensive annotation rules to educate annotators in this process. We collected approximately 10,000 comments from YouTube for all three languages. Marking the first dataset of these languages for this task, we also developed a baseline model with pre-trained transformers."
}
```
---

## 🙏 Acknowledgements

This work was conducted with the financial support of the **Science Foundation Ireland Centre for Research Training in Artificial Intelligence** under Grant No. **18/CRT/6223**, supported in part by a research grant from the Science Foundation Ireland (SFI) under Grant Number **SFI/12/RC/2289_P2** (Insight_2), and also supported by the **College of Science and Engineering, University of Galway**.

---

<div align="center">

Made with ❤️ for safer, more inclusive online spaces

**Kumaresan et al. 2024 · LREC-COLING**

</div>
