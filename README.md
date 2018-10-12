# ATL_public
Generate the Asteriseismic Target List for TESS (Schofield et al. (2018))

ATL/generateATL.py
Generate the ATL using the DR2 and XHIP catalogues, or using the TGAS and XHIP catalogues.
Calculate a detection probability, Pdet, for each star. Output: ATL_top25k.csv

ATL/DR.py
All functions needed to generate the ATL

catalogues/
The DR2 and XHIP catalogues used to make the ATL in generateATL.py
(also contains TGAS catalogue, used to make the first version of the ATL)
The files are too large to upload to Github. Instead, download them at
https://figshare.com/s/e62b08021fba321175d6
and save them in ATL_public/catalogues/ before using generateATL.py

Note: the 'full' ATL has over 300k entries, and can be downloaded at
https://figshare.com/s/aef960a15cbe6961aead 
