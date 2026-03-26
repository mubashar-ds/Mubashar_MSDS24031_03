Deep Learning Assignment 03: 
Embedding model using ResNet50 pretrained on ImageNet as backbone

---

0. Project Structure

- Codes/ all python scripts
- Dataset/ place dataset here (not included in repo)
- Saved_Models/ trained model checkpoints (not uploaded to repo)
- Embeddings/ saved embeddings (.npy) (not uploaded to repo)
- Graphs/ loss plots and retrieval results
- Training_Logs/ training logs

---

1. Dataset Setup

Download Caltech-101 dataset and place it as:

Dataset/caltech-101/

(Each class should be in separate folder --- ImageFolder format)

---

2. Install Requirements

pip install -r requirements.txt

---

3. Training

Navigate to Codes directory:
   
cd Codes

Run training for each mode using:

python train.py --data_path ../Dataset/caltech-101 --mode contrastive --epochs 5
python train.py --data_path ../Dataset/caltech-101 --mode triplet --epochs 5
python train.py --data_path ../Dataset/caltech-101 --mode hard --epochs 5

Models will be saved in:
Saved_Models/

---

4. Save Embeddings

Navigate to Codes directory:
   
cd Codes

Generate embeddings for train/validation/test using:

python save_embeddings.py --data_path ../Dataset/caltech-101 --model_path ../Saved_Models/contrastive_best.pth --mode contrastive
python save_embeddings.py --data_path ../Dataset/caltech-101 --model_path ../Saved_Models/triplet_best.pth --mode triplet
python save_embeddings.py --data_path ../Dataset/caltech-101 --model_path ../Saved_Models/hard_best.pth --mode hard

Embeddings will be saved in:
Embeddings/

---

5. Retrieval Evaluation

Navigate to Codes directory:
   
cd Codes

Compute Recall_at_K & visualize results using:

python retrieval.py --data_path ../Dataset/caltech-101 --mode contrastive
python retrieval.py --data_path ../Dataset/caltech-101 --mode triplet
python retrieval.py --data_path ../Dataset/caltech-101 --mode hard

Outputs:
- Recall_at_1, Recall_at_5
- Retrieval visualization (query + top 5)
- t-SNE plots

---

6. Inference 

Navigate to Codes directory:
   
cd Codes

Generate embedding for new images:

python inference.py --model_path ../Saved_Models/contrastive_best.pth --image_path 101/airplanes/image_0001.jpg

---