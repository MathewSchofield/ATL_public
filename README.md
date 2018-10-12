# ATL_public
Generate the Asteriseismic Target List for TESS (Schofield et al. (2018))

ATL/generateATL.py
Generate the ATL using the DR2 and XHIP catalogues, or using the TGAS and XHIP catalogues.
Calculate a detection probability, Pdet, for each star.

ATL/DR.py
All functions needed to generate the ATL

sigPdet1.py
Calculate the uncertainty on the detection probability, Pdet

prepare_catalogues/
The code and files used to prepare the TGAS and DR2 catalogues,
before the catalogues are given to generateATL.py

catalogues/
The DR2 and XHIP catalogues used to make the ATL in generateATL.py
(also contains TGAS catalogues, used to make the first version of the ATL)
