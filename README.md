# AMmRNA-BBC: Attention-Based Model with Balanced Bagging Classifier for Precise mRNA Subcellular Localization Prediction
A novel multi-stage AMmRNA-BBC model is introduced with a novel approach to biological data and their combination. In the first stage, a feature extraction method based on multimodal and innovative features such as forward and reverse alignment-based nucleotide pairs (NucP), mRNA subcellular abundance-position rate in the k-nucleotide subrange (kNF-PR), and protein-protein interaction (PPI) data is proposed. Considering the challenge of the large number of extracted features, a hybrid deep network based on the attention mechanism based on Self-Attention and Cross-Attention is proposed to map features into a separable space and extract relationships between effective features and in-situ patterns. Imbalance of samples in biological data based on mRNA location positions is one of the main challenges in this field. In this regard, in the next step, a combined method of Random Oversampler and BBC classifier is presented for accurate and robust classification against data imbalance. The proposed method is evaluated on Cefra-Seq and RNALocate datasets and the results show an improvement in the performance of the proposed method compared to state-of-the-art methods.

# Requirements
The code was run and tested on:
```
python==3.9.21
biopython==1.85
Keras==2.3.1
numpy==1.26.4
tensorflow-gpu==2.10.0
scikit-learn==1.6.1
```
install the requirements by `pip install -r requirements.txt`

# Running AMmRNA-BBC
To run AMmRNA-BBC, follow the steps below:

1. Unzip the `cefra-seq`, `RNALocate`, and `PPI_Data` files into the `Data/` directory.
2. Make sure all required information is placed correctly inside the `Data/` directory.
3. Navigate to the `Scripts/` directory and run `pre_data_features.py` to generate the input features.
4. Finally, run `main.py` to start the training or evaluation process.

# Contact
contact by github or mojtaba.zolfi.244@gmail.com
