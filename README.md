# aes148-shelving-filter

## Paper

This repository accompanies the open access paper contribution

Frank Schultz, Nara Hahn, Sascha Spors (2020):
"**Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth**".
In: *Proc. of 148th AES Convention*, Virtual Vienna, May 2020, Paper 10339,
http://www.aes.org/e-lib/browse.cfm?elib=20756

![low-shelving-filter-varying-slope.png](graphics/low-shelving-filter-varying-slope.png "Shelving Filter with +-12 dB Level and Different Slopes Built from
a Shelving Filter Cascade")

The plot shows the frequency responses of higher order shelving filters with
+-12 dB level and different target slopes built from a cascade of 2nd order
infinite impulse response shelving filters. Number of required biquads here
is chosen as 3 times the target bandwidth in octaves.

This paper was presented as remote talk at
[AES Virtual Vienna](https://www.eventscribe.com/2020/VirtualVienna/ajaxcalls/PresentationInfo.asp?efp=WUpJR1ZBVEExMjg0OQ&PresentationID=730312&rnd=0.6913832).

## Abstract

A shelving filter that exhibits an adjustable transition band is derived from a
cascade of second order infinite impulse response shelving filters.
Two of three parameters, i.e. shelving level, transition slope and
transition bandwidth, can be freely adjusted in order to describe the design
specifications.
The accuracy of the resulting response depends on the number of
deployed biquads per octave.
If this is set too small, deviations in level and bandwidth as well as a rippled
slope can occur.
The shelving filter cascade might be used in applications,
that require a fractional-order slope in a certain bandwidth,
such as for sound reinforcement system equalization, sound field synthesis
and audio production.


## Repository Content
- rendered paper, talk slides as pdf
- subfolders
  - graphics: folder for rendered graphics used in the paper and slides
  - paper: tex/tikz/bib source code for the paper
  - python: source code for the filter design, `mkfig-xxx.py` scripts render
  graphics used in paper and slides, `util.py` includes the signal processing
  stuff
  - slides: tex source code for the talk using `beamer.cls`


## Copyright

The material is licensed under

- Creative Commons Attribution 4.0 International License for rendered
text and graphics
- MIT License for source code

Please attribute the material as
Frank Schultz, Nara Hahn, Sascha Spors, University of Rostock,
Shelving Filter Cascade with Adjustable Transition Slope and
Bandwidth-Accompanying Material,
https://github.com/spatialaudio/aes148-shelving-filter,
commit number / version tag / DOI, year.
