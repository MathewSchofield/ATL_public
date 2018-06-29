# ATL_public
Generate the Asteriseismic Target List for TESS

ATL/ATL.py
Generate the ATL using XHIP, TGAS and/or DR2 catalogues.
Calculate a detection probability, Pdet, for each star.

ATL/DR.py
All functions needed to generate the ATL

sigPdet1.py
Calculate the uncertainty on the detection probability, Pdet

prepare_catalogues/
The code and files used to prepare the TGAS and DR2 catalogues,
before they are given to ATL.py
