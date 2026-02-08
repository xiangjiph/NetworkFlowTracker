# NetworkFlowTracker

## Overview
*NetworkFlowTracker* (NFT) is a Python package for particle tracking in spatial networks. It is developed for tracking red blood cells (RBCs) circulating in 3D vascular networks, where RBC densities and flow speeds vary by orders of magnitude across vessels. NFT utilizes a spatial graph representation of the network for both candidate selection and predictive tracking. When applied to experimental data, NFT achieves high tracking accuracy and increases the maximum trackable speed of RBCs by 2 to 3 folds compared to conventional particle tracking algorithms.

## Requirements
- Python >=3.9 (developed and tested with 3.9.16)
- See runtime dependencies in [pyproject.toml](pyproject.toml)

## Installation
- From source:
```bash
pip install .
```
- With notebook extras:
```bash
pip install .[notebook]
```

## Data
Figure data associated with the publication has been deposited at [zenodo](https://zenodo.org/records/18110466). 

## Reference
If you use `NetworkFlowTracker`, please consider citing the following paper:

    @article{Ji2026,
       title={Microvascular architecture and physiological fluctuations constrain the control of cerebral microcirculation},
       author={Ji, Xiang and Zhao, Yuchen and Bai, Lu and Wang, Kai and Kleinfeld, David}, 
       journal={Proceedings of the National Academy of Sciences of the United States of America},
       volume={123},
       number={3},
       pages={e2521872123},
       year={2026},
       url={https://www.pnas.org/doi/abs/10.1073/pnas.2521872123}, 
       doi = {10.1073/pnas.2521872123},}

## License
BSD 3-Clause. See [LICENSE](LICENSE).