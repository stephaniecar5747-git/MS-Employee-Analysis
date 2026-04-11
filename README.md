## 🛠️ Installation & Setup

To get the pipeline running locally, follow these steps:

1. **Clone the repository:**
   *bash
   git clone [https://github.com/stephaniecar5747-git/proyecto_challenge](https://github.com/stephaniecar5747-git/proyecto_challenge)
   cd your-repo

2. **Install Python dependencies:**
    *bash
    pip install -r requirements.txt

3. **Download Language Models (SpaCy):**
    Required for text lemmatization and cleaning.

    *bash
    python -m spacy download en_core_web_sm
    python -m spacy download es_core_news_sm

4. **Setup Web Scraping (Playwright):**
    Installs the necessary browser engines.

    *bash
    playwright install

5. **Run the Pipeline:**
    *bash
    python src/mlops_pipeline.py