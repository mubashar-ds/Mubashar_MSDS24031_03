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

3. Subset Dataset Indices Making

Run this command to make a consistent subset to use throughout project:

cd Codes
python data_subset.py --data_path ../Dataset/caltech-101

4. Training

Run training for each mode using:

cd Codes
python train.py --data_path ../Dataset/caltech-101 --mode contrastive --epochs 5
python train.py --data_path ../Dataset/caltech-101 --mode triplet --epochs 5
python train.py --data_path ../Dataset/caltech-101 --mode hard --epochs 5

Models will be saved in:
Saved_Models/

---

5. Save Embeddings

Generate embeddings for train/validation/test using:

cd Codes
python save_embeddings.py --data_path ../Dataset/caltech-101 --model_path ../Saved_Models/contrastive_best.pth --mode contrastive
python save_embeddings.py --data_path ../Dataset/caltech-101 --model_path ../Saved_Models/triplet_best.pth --mode triplet
python save_embeddings.py --data_path ../Dataset/caltech-101 --model_path ../Saved_Models/hard_best.pth --mode hard

Embeddings will be saved in:
Embeddings/

---

6. Retrieval Evaluation

Compute Recall_at_K & visualize results using:

cd Codes
python retrieval.py --data_path ../Dataset/caltech-101 --mode contrastive
python retrieval.py --data_path ../Dataset/caltech-101 --mode triplet
python retrieval.py --data_path ../Dataset/caltech-101 --mode hard

Outputs:
- Recall_at_1, Recall_at_5
- Retrieval visualization (query + top 5)
- t-SNE plots

---

7. Inference 

Generate embedding for new images:

cd Codes
python inference.py --model_path ../Saved_Models/contrastive_best.pth --image_path ../Dataset/caltech-101/airplanes/image_0001.jpg
python inference.py --model_path ../Saved_Models/triplet_best.pth --image_path ../Dataset/caltech-101/airplanes/image_0001.jpg
python inference.py --model_path ../Saved_Models/hard_best.pth --image_path ../Dataset/caltech-101/airplanes/image_0001.jpg

For checking embeddings of two different images, along with cosine score:

cd Codes
python inference.py --model_path ../Saved_Models/contrastive_best.pth --image_path ../Dataset/caltech-101/airplanes/image_0001.jpg ../Dataset/caltech-101/airplanes/image_0002.jpg 
python inference.py --model_path ../Saved_Models/triplet_best.pth --image_path ../Dataset/caltech-101/airplanes/image_0001.jpg ../Dataset/caltech-101/airplanes/image_0002.jpg 
python inference.py --model_path ../Saved_Models/hard_best.pth --image_path ../Dataset/caltech-101/airplanes/image_0001.jpg ../Dataset/caltech-101/airplanes/image_0002.jpg 



---