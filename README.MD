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
Figure data associated with the publication has been deposited at [zenodo](https://zenodo.org/records/18110466?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjkwOWRlYTNiLTk0MmUtNGZlZi04NjExLTg5ZGU3MTZkODMxNCIsImRhdGEiOnt9LCJyYW5kb20iOiI2NjliNTdkZWE5NTljZTI0ZjA5ODNhNTUzYTMzOThlZiJ9.6wQqW4_jDWfVJDBa2tmrDQg4L5dMOHXH29UV6NCRTr5C0nlytbqv2dETxGGmdnp_lgwGXkbLvijrdGPGdoOW0Q). 

## Reference
If you use `NetworkFlowTracker`, please consider citing the following paper:

    @article{Ji2026,
       title={Microvascular architecture and physiological fluctuations constrain the control of cerebral microcirculation},
       author={Ji, Xiang and Zhao, Yuchen and Bai, Lu and Wang, Kai and Kleinfeld, David}, 
       journal={Proceedings of the National Academy of Sciences of the United States of America},
       volume={},
       number={},
       pages={},
       year={2026},
       publisher={},
       url={}}

## License
BSD 3-Clause. See [LICENSE](LICENSE).