<div align="center">

<br/>

```
 /$$    /$$                      /$$                         /$$    /$$ /$$                        
| $$   | $$                     | $$                        | $$   | $$|__/                        
| $$   | $$ /$$$$$$   /$$$$$$$ /$$$$$$    /$$$$$$   /$$$$$$ | $$   | $$ /$$  /$$$$$$  /$$  /$$  /$$
|  $$ / $$//$$__  $$ /$$_____/|_  $$_/   /$$__  $$ /$$__  $$|  $$ / $$/| $$ /$$__  $$| $$ | $$ | $$
 \  $$ $$/| $$$$$$$$| $$        | $$    | $$  \ $$| $$  \__/ \  $$ $$/ | $$| $$$$$$$$| $$ | $$ | $$
  \  $$$/ | $$_____/| $$        | $$ /$$| $$  | $$| $$        \  $$$/  | $$| $$_____/| $$ | $$ | $$
   \  $/  |  $$$$$$$|  $$$$$$$  |  $$$$/|  $$$$$$/| $$         \  $/   | $$|  $$$$$$$|  $$$$$/$$$$/
    \_/    \_______/ \_______/   \___/   \______/ |__/          \_/    |__/ \_______/ \_____/\___/ 
                                                                                                   
```

### _View the world through vectors._

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP-412991?style=for-the-badge&logo=openai&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-00BF63?style=for-the-badge&logo=pinecone&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas_Search-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

<br/>

> **Multi-modal vector search engine — image similarity, zero-shot KNN classification, and text-to-image retrieval powered by CLIP embeddings.**

<br/>

---

</div>


## ✦ What is VectorView?

VectorView is a **multi-modal vector search system** that converts raw images and natural language into high-dimensional embeddings using OpenAI's **CLIP model** — then stores, indexes, and retrieves them via **Pinecone** and **MongoDB Atlas Vector Search**.

No supervised training. No labeled pipelines. Just embeddings and geometry.

```
You give it: "A Running Dog"
It returns: 🐶🐕— the closest matching images in the dataset
```

<br/>

---
 
## ✦ Architecture
 
```
┌─────────────────────────────────────────────────────────────────┐
│                       VectorView Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   📁 Animal Dataset (cats / dogs / wolves)                      │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────┐    CLIP Model (ViT-B/32)                      │
│   │   Image     │ ──────────────────────────► [512-d vector]    │
│   │   Encoder   │                                    │          │
│   └─────────────┘                                    │          │
│                                                      ▼          │
│   ┌─────────────┐                         ┌──────────────────┐  │
│   │   Text      │ ──► [512-d vector] ────►│  Vector Database │  │
│   │   Encoder   │                         │                  │  │
│   └─────────────┘                         │  ┌────────────┐  │  │
│                                           │  │  Pinecone  │  │  │
│   ┌──────────────────────────────────┐    │  └────────────┘  │  │
│   │           Query                  │    │  ┌────────────┐  │  │
│   │  "cute Kitty" / query_image.jpg  │    │  │  MongoDB   │  │  │
│   └──────────────────────────────────┘    │  └────────────┘  │  │
│             │                             └──────────────────┘  │
│             ▼                                      │            │
│        Cosine Similarity  ◄────────── Top-K Results (k=5)       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
 
<br/>

---
 
## ✦ Tasks
 
<summary><b>🔍 Task 1 — Image Similarity Search</b></summary>
<br/>
**Goal:** Given a query image, retrieve the most visually similar images from the dataset.
 
**How it works:**
1. Embed the query image with CLIP into a 512-d vector
2. Send the vector to Pinecone
3. Pinecone computes **cosine similarity** and returns the top-K nearest neighbors
4. Display the results alongside similarity scores

**Output:**
<img width="1911" height="577" alt="Screenshot 2026-06-17 150443" src="https://github.com/user-attachments/assets/3b6ad443-2c7a-451e-9f9e-4c1160b61ebd" />
<img width="1892" height="418" alt="image" src="https://github.com/user-attachments/assets/95a6da1e-f61c-4782-acbd-ff26c4c3de18" />


<summary><b>🏷️ Task 2 — Zero-Shot KNN Classification</b></summary>
<br/>
**Goal:** Classify an unknown image without ever training a classifier.
 
**How it works:**
1. Embed the query image with CLIP
2. Retrieve the top-K nearest neighbors from Pinecone
3. Run a **majority vote** on neighbor labels
4. The predicted class is the most common label among neighbors — zero training required

**Output:**
<img width="1390" height="706" alt="image" src="https://github.com/user-attachments/assets/4001723f-4545-4673-84f4-1ea57139e749" />
<img width="1101" height="593" alt="image" src="https://github.com/user-attachments/assets/b9846aac-a331-4080-b298-f7fa83a7e9a2" />




<summary><b>💬 Task 3 — Cross-Modal Search (Text → Image)</b></summary>
<br/>
**Goal:** Search the image dataset using natural language queries.
 
**How it works:**
1. Encode a text prompt (e.g. `"wolf in snow"`) with CLIP's text encoder
2. The text vector lands in the **same embedding space** as the images
3. MongoDB Atlas `$vectorSearch` finds the nearest image vectors
4. Return and display the top matching images

**Output:**
<img width="1448" height="517" alt="image" src="https://github.com/user-attachments/assets/f813c482-3e14-4d9c-889c-62e6f3c83ca4" />
<img width="1476" height="533" alt="image" src="https://github.com/user-attachments/assets/a22dc9c0-0d9b-4368-a36f-f4f02782ccca" />
<img width="1482" height="520" alt="image" src="https://github.com/user-attachments/assets/68aeaac4-8a74-40e6-9fbf-c38540d619bb" />
<img width="1482" height="487" alt="image" src="https://github.com/user-attachments/assets/01a574c0-c94e-4fe0-a25a-12f7640eea10" />
<img width="1488" height="547" alt="image" src="https://github.com/user-attachments/assets/3d85e56c-4b05-4d4b-93da-110c344b00a8" />
<img width="1490" height="541" alt="image" src="https://github.com/user-attachments/assets/2b9720cc-f8d4-4723-96aa-5808534de765" />



---

## ✦ Tech Stack
 
| Component | Tool | Role |
|---|---|---|
| Embedding Model | `openai/clip-vit-base-patch32` | Image + Text → 512-d Vector |
| Vector DB (Primary) | Pinecone (Serverless) | Hosted ANN search, Tasks 1 & 2 |
| Vector DB (Comparison) | MongoDB Atlas | Vector search via `$vectorSearch`, Task 3 |
| Runtime | Python 3.10+, Google Colab | Development environment |
| Visualization | Matplotlib | Display query + result images |
 
<br/>

---

## ✦ Setup
 
### 1. Clone the repo
 
```bash
git clone https://github.com/your-username/vector-view.git
cd vector-view
```
 
### 2. Install dependencies
 
```bash
pip install torch torchvision transformers pinecone pymongo[srv] Pillow matplotlib tqdm
```
 
### 3. Prepare the dataset
 
Organize your images like this:
 
```
vector-view/
└── animal_dataset/
    ├── cats/
    │   ├── cat_001.jpg
    │   └── ...
    ├── dogs/
    │   ├── dog_001.jpg
    │   └── ...
    └── wolves/
        ├── wolf_001.jpg
        └── ...
```
 
> In Colab: upload `animal_dataset.zip` via **Files → Upload**, then run the extraction cell.
 
### 4. Configure API credentials
 
Open the **Configuration** cell in the notebook and fill in your keys:
 
```python
PINECONE_API_KEY  = "your_pinecone_key"      # app.pinecone.io
PINECONE_ENV      = "us-east-1"              # your Pinecone region
PINECONE_INDEX    = "animal-embeddings"      # auto-created
 
MONGO_URI         = "mongodb+srv://..."      # MongoDB Atlas URI
MONGO_DB          = "your_database"
MONGO_COLLECTION  = "animal_images"
```
 
### 5. Run the notebook
 
Execute cells top-to-bottom in Colab or Jupyter. Each task is self-contained after setup.
 
<br/>

---

## ✦ Key Concepts
 
**Why CLIP?**
CLIP (Contrastive Language–Image Pretraining) was trained to align image and text representations in a shared vector space. This means `encode("a dog")` and `encode(dog_image.jpg)` produce geometrically close vectors — enabling cross-modal retrieval with **zero task-specific training**.
 
**Why Vector Databases?**
Traditional SQL `WHERE` clauses can't find *similar* things. Vector DBs use Approximate Nearest Neighbor (ANN) algorithms — like HNSW or IVFFlat — to find the closest vectors in milliseconds, even across millions of entries.
 
**Cosine Similarity:**
```
similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
```
Range `[-1, 1]` → closer to `1` = more similar.
 
<br/>

---
 
## ✦ Pinecone vs. MongoDB Atlas
 
| Feature | Pinecone | MongoDB Atlas |
|---|---|---|
| Type | Purpose-built vector DB | Multi-model DB + vector search |
| Setup | Fully managed, API-first | Atlas cluster + index config |
| Query Speed | Extremely fast (dedicated ANN) | Fast, but secondary to document ops |
| Metadata Filtering | ✅ Native | ✅ via `$match` pipeline |
| Best For | Pure vector workloads | Apps needing vectors + documents |
 
<br/>

---
 
## ✦ Observations
 
- CLIP embeddings are **surprisingly robust** — text queries like `"running dog"` retrieve contextually correct images with no fine-tuning
- KNN classification via vector search **matches simple CNN accuracy** on clean datasets with zero training overhead
- Pinecone offers faster cold-start indexing; MongoDB is preferable when your app already stores structured documents
- Cross-modal search degrades on **ambiguous text** — `"animal"` returns scattered results across all classes

<br/>

---
 
## ✦ Project Structure
 
```
VectorView/
├── VectorView.ipynb            # Main notebook
├── animal_dataset/             # Image dataset (not tracked in git)
├── screenshots/                # Output screenshots for README
└── README.md
```
 
<br/>

---

<div align="center">

*A View From Vectors*

[![GitHub](https://img.shields.io/badge/GitHub-abdullahwaheed0112-181717?style=flat-square&logo=github)](https://github.com/abdullahwaheed0112)
 
</div>
