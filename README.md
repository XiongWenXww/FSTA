<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> Brain Effective Connectivity Estimation via Fourier Spatiotemporal Attention (KDD 2025, Best Student Paper Runner-Up) </b></h2>
</div>

## Introduction of output:

The code is executed 20 times, with each run comprising 300 epochs. In order to have outputs for the 0th and 300th epochs, so the epoch is set to 301.  During the code execution, the loss, 5 evaluation metrics, and the corresponding threshold value of the obtained brain effective connectivity matrix are outputted every 100 epochs. 
The threshold value is used to binarize the matrix. Additionally, both the original brain effective connectivity matrix and the binarized version are also generated as outputs.

## Implications of the output brain effective connectivity matrix:

The main point to note is that, since the value of the i-th row and j-th column of the ground truth refers to brain region vi exerting a causal effect on brain region vj, 
and the brain effective connectivity matrix obtained by the FSTA-EC indicates the opposite meaning. In the code, we transposed the brain effective connectivity matrix obtained by the FSTA-EC to obtain the final output brain effective connectivity matrix. 
Let the output brain effective connectivity matrix be A, then Aij indicates that brain region vi exerts a causal effect on brain region vj.

## Download dataset
- **Simulated fMRI datasets:** You can access the well simulated fMRI datasets from [[Sanchez]](https://github.com/cabal-cmu/feedbackdiscovery).

- **Real resting-state fMRI dataset:** You can access the real resting-state fMRI dataset we used by navigating to the [[Real]](https://github.com/shahpreya/MTlnet) and placing it in the folder ... /fMRI/.

## Quick Start:

```
python train_FSTA_sanch_multi.py
```

By default, the code utilizes the Sim1 dataset. If you wish to change the dataset, simply modify the "index" parameter in the main() function settings within the "train_FSTA_sanch_multi.py" file.

Train the model in real fMRI dataset:

```
python train_FSTA_real_multi.py
```

Similarly, the default dataset in the code corresponds to the fMRI dataset of the left hemisphere medial temporal lobe. 
If you intend to switch to the fMRI dataset of the right hemisphere medial temporal lobe, adjust the parameter setting in the main() function of the "train_FSTA_real_multi.py" file from "pos" to "right".
