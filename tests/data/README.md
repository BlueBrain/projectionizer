### [PH]y.nrrd, [PH]6.nrrd, brain_regions.nrrd

The test data was generated with the following script:

```python
import numpy as np
from voxcell import VoxelData

S = 30
L = 62

# phy is (S,L,S) shaped. values in Y dimension goes from roughly -1 to 61
# with the plane Y==1 being filled with exactly 0s
phy = np.zeros((S, S, L))
phy[..., :] = np.arange(-1, L-1)
phy += np.random.random(phy.shape)
phy[..., 1] = 0
phy = phy.transpose(1, 2, 0)

ph6 = np.full((*phy.shape, 2), np.nan)
br = np.zeros(phy.shape)

# each layer has its own region id [11,...,16]
for i in range(6):
    ind = 51 - i*10
    br[1:-1, ind:ind+10, 1:-1] = 11+i

# L6 low and upper limits are 0 and 10 along the whole L6 region
ph6[1:-1, ind:ind+10, 1:-1] = [0, 10]

vdy = VoxelData(phy, [1]*3)
vdy.save_nrrd('./[PH]y.nrrd')
vdy.with_data(br).save_nrrd('./brain_regions.nrrd')
vdy.with_data(ph6).save_nrrd('./[PH]6.nrrd''')
```

The region IDs and region names can be found in `hierarchy.json`.
