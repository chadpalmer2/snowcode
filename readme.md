# SnowCode

As a senior project in Computer Science and Mathematics at Yale, I designed and implemented a two-dimensional matrix barcode, which I have named SnowCode.

## Abstract

Two-dimensional matrix barcodes, such as the now ubiquitous QR code, see widespread usage across industry as a means to store data in an image format for later recovery by an optical sensor or camera. Having an interest in visual representation of data, I chose to design and implement my own such barcode, with the goal of prioritizing the aesthetic appearance of the barcode while preserving storage and data robustness requirements.

Through consideration of various designs, I settled on a design emulating that of a snowflake, storing information in the varying lengths of lines and utilizing hexagonal shapes for image detection and transformation. This code, which I named SnowCode, featured a storage capacity of 108 bytes, 27 of which were set aside for Reed-Solomon error correction, an algebraic code which adds a number of error correction symbols to efficiently introduce redundancy and improve recovery of the original data.

I implemented an image encoder and decoder for this design in Python 3, utilizing a number of libraries for error correction, image manipulation, and computer vision. The resulting implementation featured both a command line interface with a debugging mode, and a deployed front-end web application for use by both desktop and mobile devices.

The program achieved success in encoding and decoding similar to the prototypical QR code, and could efficiently encode data and decode images in the presence of skewing, rotation, noise, and physical damage to the SnowCode. However, the program would occasionally encounter errors in the presence of image noise and other image quality issues, which would result in improper hexagon detection and image transformation. Despite this, the program was a successful implementation of the design, and provided a number of avenues for incremental improvement.

## Deployed version

A deployed version of the frontend Flask app can be found at <https://snowcode-chad.herokuapp.com/>.

## Installation

After pulling to your local device, run

```
pip install -r requirements.txt
```

The `snowcode.py` module can be run in terminal with `python3 snowcode.py`, which provides a simple CLI and access to a debugging mode.

The frontend Flask app can be launched by running `python3 -m flask run`.