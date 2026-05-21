# Clinical Decision Support System — MRI Brain Tumor Diagnosis

A multi-agent AI system that assists radiologists in diagnosing brain tumors from MRI scans. A ResNet18 CNN classifies the tumor type, a mandatory human-in-the-loop checkpoint ensures radiologist oversight, and a RAG-powered clinical report agent generates a structured medical report.

---

## Architecture

```
MRI Upload → [Agent 1: Image Classifier (ResNet18)] → Confidence Check
                                                           ↓
                                              [HITL: Radiologist Review]
                                                    Approve / Reject / Override
                                                           ↓ (if approved)
                                          [Agent 2: Clinical Report (RAG + Template)]
                                                           ↓
                                                    PDF + JSON Report
```

**4 tumor classes:** glioma · meningioma · pituitary · no\_tumor

---

## Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd clinical-decision-support-system

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env: add your GEMINI_API_KEY (optional), verify MODEL_PATH
```

### 3. Train the model (Google Colab recommended)

```bash
# Download dataset
pip install kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/raw/

# Train (or open notebooks/02_model_training.ipynb in Colab)
python -m src.models.train \
    --data_dir data/raw/Training \
    --val_dir  data/raw/Testing  \
    --epochs   10               \
    --output   src/checkpoints/resnet18_braintumor.pt
```

### 4. Build the RAG index

```python
from src.tools.rag_retrieval_tool import RAGRetrievalTool
rag = RAGRetrievalTool()
rag.build_index()
```

Or run `notebooks/04_rag_setup.ipynb`.

### 5. Run the pipeline

```bash
# Interactive mode (radiologist reviews at terminal)
python -m src.main --image path/to/mri.jpg --patient_id PATIENT_001

# Non-interactive mode (auto-approve HITL — for demos/tests)
python -m src.main --image path/to/mri.jpg --non_interactive
```

### 6. Run the automated demo

```bash
# Copy 4 sample MRI images (one per class) to demo/sample_mris/
python demo/demo_script.py
```

---

## Evaluate the Model

```bash
python -m src.models.evaluate \
    --test_dir data/raw/Testing \
    --model    src/checkpoints/resnet18_braintumor.pt \
    --output   model_evaluation.json
```

Generates `model_evaluation.json` + `confusion_matrix.png`.

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
clinical-decision-support-system/
├── src/
│   ├── main.py                        # Entry point
│   ├── config.py                      # Configuration & constants
│   ├── logger.py                      # JSON logging
│   ├── agents/
│   │   ├── image_classifier_agent.py  # Agent 1: CNN classification
│   │   ├── clinical_report_agent.py   # Agent 2: Report generation
│   │   ├── orchestrator_agent.py      # Pipeline coordination
│   │   └── base_agent.py
│   ├── tools/
│   │   ├── cnn_inference_tool.py      # ResNet18 wrapper
│   │   ├── rag_retrieval_tool.py      # ChromaDB RAG
│   │   ├── report_formatter_tool.py   # PDF/JSON export
│   │   └── error_handlers.py
│   ├── models/
│   │   ├── cnn_model.py               # ResNet18 definition
│   │   ├── preprocessing.py           # Image preprocessing
│   │   ├── inference.py               # Forward-pass wrapper
│   │   ├── train.py                   # Training script
│   │   └── evaluate.py                # Evaluation script
│   ├── data/
│   │   ├── medical_guidelines.txt     # Clinical corpus for RAG
│   │   └── class_descriptions.json
│   ├── checkpoints/                   # Trained model weights
│   ├── vector_store/                  # ChromaDB index
│   └── hitl/
│       └── review_interface.py        # Radiologist review UI
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_model_training.ipynb        # Colab-friendly
│   ├── 03_model_evaluation.ipynb
│   └── 04_rag_setup.ipynb
├── tests/
│   ├── test_image_preprocessing.py
│   ├── test_cnn_model.py
│   ├── test_rag_retrieval.py
│   └── test_agents.py
├── demo/
│   ├── demo_script.py
│   └── sample_mris/
├── logs/
│   ├── runs/                          # Per-run JSON logs
│   └── hitl_decisions.json            # Audit trail
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MODEL_DETAILS.md
│   ├── RAG_SETUP.md
│   └── MEDICAL_BACKGROUND.md
├── requirements.txt
└── .env.example
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| ResNet18 over EfficientNet/ViT | Best speed-accuracy tradeoff; proven for medical imaging at this scale |
| HITL between Agent 1 and Agent 2 | Medically essential — no report should come from an unvalidated AI prediction |
| Confidence threshold 0.80 | Flags ~5% of ambiguous cases; keeps HITL reviews focused |
| ChromaDB for RAG | Embedded, no external server, Python-native |
| JSON logging everywhere | Full audit trail required for clinical AI systems |
| Static fallback for RAG | Pipeline never fails completely due to retrieval issues |

---

## Dependencies

Core: `torch`, `torchvision`, `Pillow`, `chromadb`, `sentence-transformers`, `reportlab`  
Dev: `pytest`, `numpy`, `matplotlib`, `scikit-learn`

See `requirements.txt` for pinned versions.

---

## License

For educational and research use.  
**This system is NOT a certified medical device. Always consult a qualified physician for diagnosis.**
