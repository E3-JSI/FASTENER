# FASTENER
___FeAture SelecTion ENabled by EntRopy___

FASTENER is a state-of-the-art feature selecton algorithm for remote sensing, but performs well also on several other data sets. It is most suitable for large datasets with several hundreds of features. It has been develped for the use case of crop/land-cover classification based on Sentinel-II data.

The algorithm

## Users' Manual
Basic documentation is available within the code.

## Future Work

* Create PyPI module
* Update documentation
* Prepare example notebooks
* Create unit tests

## Publications

If you use the algorithm, please cite the following paper:

* Koprivec, F.; Kenda, K.; Šircelj, B., FASTENER Feature Selection for Inference from Earth Observation Data. Entropy 2020, ?, ?. (_minor reviews pending_)

__Abstract__:

> In this paper, FASTENER feature selection algorithm is presented.
    The algorithm exploits entropy-based measures such as mutual information in the crossover phase of the genetic algorithm approach.
    FASTENER converges to an (near) optimal subset of features faster than previous state-of-the-art algorithms and achieves better classification accuracy than similarity-based methods such as KBest or ReliefF or wrapper methods such as POSS.
    The approach was evaluated using the Earth Observation dataset for land-cover classification from ESA's Sentinel-2 mission, the digital elevation model and the ground truth data of the Land Parcel Identification System from Slovenia.
    The algorithm can be used in any statistical learning scenario.


### Acknowledgements
This research was funded by European Union's Horizon 2020 programme project PerceptiveSentinel (Research and Innovation) grant number [776115](https://cordis.europa.eu/project/id/776115), project NAIADES (Innovation Action) grant number [820985](https://cordis.europa.eu/project/id/820985).