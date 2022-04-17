# lp-lim

This little program, written on 2018-06-25, is an attempt on an unusual limiter for audio. 
The idea was that with bass-heavy tracks (genres like EDM), the gain function could be timed
according to zero-crossings of the bass, which not only provides tighter limiting (the signal
will hit 0dBFS >50 times a second), but most importantly the introduced distortion would
be contained primarily within the bass frequency and its harmonics.

This program needs FFTW3: http://fftw.org/