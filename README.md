# Anomaly

Copyright 2017 Xingchen Wan
Xingchen.Wan@deutsche-boerse.com | Deutsche Boerse AG | Frankfurt am Main, Germany
Xingchen.Wan@st-annes.ox.ac.uk | Department of Engineering Science, University of Oxford | Oxford, UK


Core algorithms of the gts anomaly detector

Features an ensemble of algorithms (some of them are not implemented yet), including:

Probabilistic Exponentially-weighted Moving Average

Simple Moving Average

Grubb's Outlier Test

Kolmogorov–Smirnov Test

Conformity-based k-nearest Neighbours

First-order derivative

Median Absolute Deviation

Welch's t-test

F-test for equality of variance

Mann-Kendall Trend Test

Theil-Sen's Slope

... and finally an stochastic gradient descent (SGD) incremental support vector machine / simple majority vote assembler to turn this
collection of results into a final decision on whether a point is anomalous, and if so what kind of anomaly it is.

With acknowledgements to LinkedIn Luminol, Skyline, Twitter Anomaly Detection respositories.
